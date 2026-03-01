#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — FastAPI Backend for ISO 14224 Chatbot
รวม RAG logic จาก 14Chat_r1.py เสิร์ฟ HTML + /api/chat endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import httpx
import os
import traceback

# ============================================================
# CONFIG
# ============================================================
CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "iso14224"
HTML_FILE        = "./iso14224_chatbot.html"

# ============================================================
# INIT CHROMADB (embedded — ใช้ไฟล์จาก git)
# ============================================================
def init_collection():
    try:
        ef = embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        # รองรับ ChromaDB ทั้งเก่าและใหม่
        try:
            raw_cols = client.list_collections()
            if raw_cols and hasattr(raw_cols[0], "name"):
                names = [c.name for c in raw_cols]
            else:
                names = [str(c) for c in raw_cols]
        except Exception:
            names = []

        if COLLECTION_NAME not in names:
            col = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
        else:
            col = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        return col

    except Exception as e:
        err = str(e)
        if "no such column" in err or "schema" in err.lower():
            import shutil
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
            ef = embedding_functions.DefaultEmbeddingFunction()
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            return client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
        raise

# collection จะถูก init ใน lifespan หลัง server ขึ้น
collection = None

# ============================================================
# FASTAPI APP — ใช้ lifespan เพื่อให้ server ขึ้นก่อน แล้วค่อย load VDB
# ============================================================
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    global collection
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        collection = await loop.run_in_executor(pool, init_collection)
    yield

app = FastAPI(title="ISO 14224 RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST MODEL
# ============================================================
class ChatRequest(BaseModel):
    question: str
    provider: str          # "deepseek" | "anthropic" | "gemini"
    search_mode: str       # "general" | "failure_mode" | "table_codes"
    top_k: int = 15
    temperature: float = 0.3
    max_tokens: int = 1500

# ============================================================
# API KEY RESOLVER — อ่านจาก Environment Variables
# ============================================================
ENV_KEY_MAP = {
    "deepseek":  "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}

def get_api_key(provider: str) -> str:
    env_var = ENV_KEY_MAP.get(provider)
    if not env_var:
        raise HTTPException(400, f"Unknown provider: {provider}")
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise HTTPException(500, f"API key not configured on server. Set environment variable: {env_var}")
    return key
def rag_search(query: str, n: int = 15) -> list[dict]:
    if collection is None:
        return [{"text": "[VDB initializing, please retry in a moment]", "meta": {}, "score": None}]
    try:
        results = collection.query(query_texts=[query], n_results=n)
        docs   = results["documents"][0]
        metas  = results["metadatas"][0]
        dists  = results.get("distances", [[]])[0]

        contexts = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            score = round(1 - dists[i], 3) if dists else None
            contexts.append({
                "text":  doc,
                "meta":  meta,
                "score": score,
            })
        return contexts
    except Exception as e:
        return [{"text": f"[VDB Error: {e}]", "meta": {}, "score": None}]

# ============================================================
# SEARCH MODE LOGIC  (ported from 14Chat_r1.py)
# ============================================================
def build_queries(question: str, search_mode: str) -> tuple[str, str]:
    """Returns (vdb_query, llm_user_message)"""

    if search_mode == "failure_mode":
        vdb_q = f"Table B Annex B Failure modes {question} equipment class"
        llm_q = f"""ผู้ใช้ต้องการทราบรายการ Failure Mode (โหมดความล้มเหลว) ทั้งหมดของอุปกรณ์: '{question}'

CRITICAL INSTRUCTIONS FOR MATRIX EXTRACTION (STEP-BY-STEP):
STEP 1: CATEGORY IDENTIFICATION — ตรวจสอบใน Context ว่า '{question}' จัดอยู่ใน Equipment Category ใด
STEP 2: TABLE MATCHING — ค้นหา "ตาราง Matrix" ในหมวด Annex B ที่ตรงกับ Category นั้น (ตาราง B.6 ถึง B.14)
STEP 3: COLUMN TARGETING — เล็งไปที่คอลัมน์ของ '{question}' หรือ Equipment Class Code โดยเฉพาะ
STEP 4: ROW EXTRACTION (CRITICAL) — กวาดสายตาดูข้อมูลทีละแถว หากแถวใดมีเครื่องหมาย 'X' ให้ดึงข้อมูลมาทั้งหมด
STEP 5: COMPLETENESS CHECK — ห้ามข้าม ตัดทอน หรือสรุปเอาเอง ต้องดึงมาให้ครบทุกรหัสที่มีเครื่องหมาย 'X'
STEP 6: FORMATTING — แสดงผลเป็น Bullet points: [รหัส 3 ตัวอักษร] - [ชื่อ Failure Mode] : [คำอธิบาย/ตัวอย่าง]

หากค้นหาใน Context แล้วไม่พบตารางที่มีคอลัมน์ของ '{question}' ให้แจ้งว่าไม่พบข้อมูลอย่างสุภาพ"""

    elif search_mode == "table_codes":
        vdb_q = f"Table B Annex B {question} Code notation description"
        llm_q = f"""ผู้ใช้ต้องการทราบข้อมูลและรหัสทั้งหมดของตาราง: '{question}' (เช่น Failure cause, Failure mechanism, Detection method)

CRITICAL INSTRUCTIONS FOR MASTER TABLES EXTRACTION:
1. TABLE TARGETING: ค้นหาตารางใน Annex B ที่ตรงกับ '{question}' (เช่น Table B.2, B.3, B.4, B.5)
2. MULTI-PAGE MERGING: นำข้อมูลจากทุก Source มาปะติดปะต่อกันให้สมบูรณ์
3. NO TRUNCATION: ห้ามย่อ ห้ามสรุป และห้ามข้ามหมวดหมู่ใดๆ
4. FORMATTING: แสดงผลในรูปแบบ Markdown Table ประกอบด้วยคอลัมน์: [รหัส (Code)], [ชื่อ (Notation)], [คำอธิบาย (Description)]"""

    else:  # general
        vdb_q = question
        llm_q = question

    return vdb_q, llm_q

# ============================================================
# SYSTEM PROMPT  (ported from 14Chat_r1.py)
# ============================================================
def build_system_prompt(context_str: str) -> str:
    return f"""You are an elite Senior Reliability Engineer and Subject Matter Expert strictly adhering to ISO 14224:2016.

### 📜 CORE RAG DIRECTIVES:
1. FACTUAL ANCHORING: Every technical data point MUST be extracted ONLY from the provided Context. Do not hallucinate.
2. MANDATORY CITATION: Cite the source at the end of each claim using format [Source X | Page Y].
3. MISSING DATA: If Context does not contain the answer, state: "ข้อมูลอ้างอิงจากคลังข้อมูล ISO 14224 ที่ค้นพบ ไม่มีระบุเรื่องนี้ไว้อย่างชัดเจนครับ" before offering general advice.

### ⚙️ ISO 14224 SPECIFIC KNOWLEDGE:
- TAXONOMY AWARENESS: Level 6 = Equipment Unit, Level 7 = Subunit, Level 8 = Maintainable Item
- FAILURE DISTINCTION:
  * Failure Mode — HOW it failed (e.g., FTS = Failure to start)
  * Failure Mechanism — Physical/chemical process (e.g., Corrosion, Wear)
  * Failure Cause / Root Cause — WHY it happened (e.g., Design error, Operational error)
- MATRIX TABLES: If context contains matrix tables (Table B.6+), trace 'X' marks in the specific equipment column carefully.

### 🧠 FORMATTING GUIDELINES:
1. CODE EXPANSION: Always expand 3-letter codes with full description
2. EXPERT NOTE: Internal engineering knowledge (not from RAG) must be labeled as "💡 Expert Note (คำอธิบายเสริม):"
3. READABILITY: Use markdown, bold key terms, bullet points
4. LANGUAGE: Respond in the same language the user used

---
CONTEXT FROM ISO 14224 DATABASE:
{context_str}
"""

# ============================================================
# LLM CALL (async via httpx)
# ============================================================
async def call_llm(provider: str, api_key: str, system: str,
                   user_msg: str, temperature: float, max_tokens: int) -> str:
    async with httpx.AsyncClient(timeout=90) as client:

        if provider == "deepseek":
            resp = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user_msg},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        elif provider == "anthropic":
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6-20251205",
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user_msg}],
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": f"{system}\n\nQuestion: {user_msg}"}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        else:
            raise ValueError(f"Unknown provider: {provider}")

# ============================================================
# ROUTES
# ============================================================
@app.get("/")
async def serve_html():
    if os.path.exists(HTML_FILE):
        return FileResponse(HTML_FILE, media_type="text/html")
    return JSONResponse({"error": "HTML file not found"}, status_code=404)

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Resolve API key from environment variable
        api_key = get_api_key(req.provider)

        # 1. Build mode-aware queries
        vdb_query, llm_user_msg = build_queries(req.question, req.search_mode)

        # 2. Search VDB
        contexts = rag_search(vdb_query, req.top_k)

        # 3. Build context string with citations
        context_str = "\n\n---\n\n".join(
            f"[Source {i+1} | Page {c['meta'].get('page_number', '?')}] "
            f"{c['meta'].get('title', 'ISO 14224')}\n{c['text']}"
            for i, c in enumerate(contexts)
        )

        # 4. Build system prompt
        system = build_system_prompt(context_str)

        # 5. Call LLM
        answer = await call_llm(
            req.provider, api_key,
            system, llm_user_msg,
            req.temperature, req.max_tokens,
        )

        return {
            "answer":   answer,
            "contexts": contexts,
            "vdb_query": vdb_query,
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"LLM API Error: {e.response.text[:300]}")
    except Exception as e:
        raise HTTPException(500, f"Server Error: {str(e)}\n{traceback.format_exc()[:500]}")

@app.get("/api/keys/status")
async def keys_status():
    """แสดงว่า API Key ไหนถูก configure ไว้บน server (ไม่เปิดเผย key จริง)"""
    return {
        p: bool(os.environ.get(env_var, "").strip())
        for p, env_var in ENV_KEY_MAP.items()
    }

@app.get("/api/vdb/status")
async def vdb_status():
    try:
        count = collection.count()
        return {"status": "online", "collection": COLLECTION_NAME, "count": count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
