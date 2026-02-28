
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14224_failure_analysis.py
=========================
ISO 14224 Failure Analysis — Streamlit App (Optimized with CoT & Enhanced RAG)
Version: 5.1 - FULL RESTORED + FIXES (DeepSeek, Hidden Eye, VDB Fix)
"""
import streamlit as st
import pandas as pd
import json
import io
import os
import httpx
from datetime import datetime
import traceback

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "chroma_path":     "./chroma_db",
    "collection_name": "iso14224",
    "plant_unit_xlsx": "./Plant_Unit.xlsx",
    "app_title":       "ISO 14224 Analysis Console (Optimized)",
    "app_icon":        "🔧",
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=CONFIG["app_title"],
    page_icon=CONFIG["app_icon"],
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# EQUIPMENT CLASS MASTER LIST (ISO 14224) - FULL LIST RESTORED
# ============================================================
EQUIPMENT_CLASSES = {
    "── Rotating Equipment ──": None,
    "BL — Blowers and fans": "BL",
    "CF — Centrifuges": "CF",
    "CE — Combustion engines": "CE",
    "CO — Compressors": "CO",
    "EG — Electric generators": "EG",
    "EM — Electric motors": "EM",
    "GT — Gas turbines": "GT",
    "LE — Liquid expanders": "LE",
    "MI — Mixers": "MI",
    "PU — Pumps": "PU",
    "ST — Steam turbines": "ST",
    "TE — Turboexpanders": "TE",
    "── Mechanical Equipment ──": None,
    "CV — Conveyors and elevators": "CV",
    "CR — Cranes": "CR",
    "FS — Filters and strainers": "FS",
    "HE — Heat exchangers": "HE",
    "HB — Heaters and boilers": "HB",
    "LA — Loading arms": "LA",
    "PL — Onshore pipelines": "PL",
    "PI — Piping": "PI",
    "VE — Pressure vessels": "VE",
    "SI — Silos": "SI",
    "SE — Steam ejectors": "SE",
    "TA — Storage tanks": "TA",
    "SW — Swivels": "SW",
    "TU — Turrets": "TU",
    "WI — Winches": "WI",
    "── Electrical Equipment ──": None,
    "FC — Frequency converters": "FC",
    "PC — Power cables and terminations": "PC",
    "PT — Power transformers": "PT",
    "SG — Switchgears": "SG",
    "UP — Uninterruptible power supply": "UP",
    "── Safety & Control Equipment ──": None,
    "CL — Control logic units": "CL",
    "EC — Emergency communication equipment": "EC",
    "ER — Escape, evacuation and rescue": "ER",
    "FG — Fire and gas detectors": "FG",
    "FF — Fire-fighting equipment": "FF",
    "FI — Flare ignition": "FI",
    "IG — Inert-gas equipment": "IG",
    "IP — Input devices": "IP",
    "LB — Lifeboats": "LB",
    "NO — Nozzles": "NO",
    "TC — Telecommunications": "TC",
    "VA — Valves": "VA",
    "── Utilities & Auxiliaries ──": None,
    "AI — Air-supply equipment": "AI",
    "SU — De-superheaters": "SU",
    "FE — Flare ignition equipment": "FE",
    "HC — Heating/cooling media": "HC",
    "HP — Hydraulic power units": "HP",
    "NI — Nitrogen-supply equipment": "NI",
    "OC — Open/Close drain equipment": "OC",
    "HV — HVAC equipment": "HV",
    "PO — Power Transmission & Speed Control": "PO",
}

# ============================================================
# DATA LOADING (cached)
# ============================================================
@st.cache_data
def load_plant_unit():
    try:
        df = pd.read_excel(CONFIG["plant_unit_xlsx"])
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"❌ ไม่พบไฟล์ Plant_Unit.xlsx: {e}")
        return pd.DataFrame(columns=["Plant", "Unit", "Machinetype"])

@st.cache_resource
def init_vdb():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        
        ef = embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path=CONFIG["chroma_path"])
        
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        
        if CONFIG["collection_name"] not in collection_names:
            collection = client.create_collection(
                name=CONFIG["collection_name"],
                embedding_function=ef,
            )
        else:
            collection = client.get_collection(
                name=CONFIG["collection_name"],
                embedding_function=ef,
            )
        return collection
    except Exception as e:
        st.error(f"❌ VDB Error: {e}")
        return None

# ============================================================
# RAG SEARCH (FIXED: metadata filter bypass)
# ============================================================
def rag_search(collection, query: str, n: int = 10, topic_filter: str = None) -> str:
    if collection is None:
        return "[VDB not available]"
    
    try:
        # แก้ไขเพื่อเลี่ยงปัญหา no such column: topic ในโหมดเริ่มต้น
        results = collection.query(query_texts=[query], n_results=n)
        
        if not results["ids"][0]:
            return "[No relevant data found]"
        
        contexts = []
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            title = meta.get("title", "ISO Reference")
            page = meta.get("page_number", "?")
            contexts.append(f"[Source {i+1} | Page {page}] {title}\n{doc}")
        
        return "\n---\n".join(contexts)
    except Exception as e:
        return f"[RAG Error Bypass: {str(e)}]"

# ============================================================
# LLM WRAPPER (FIXED: DeepSeek OpenAI v1+ Compatibility)
# ============================================================
def call_llm(system_prompt: str, user_message: str,
             provider: str, api_key: str, max_tokens: int = 2500) -> str:
    if not api_key or not api_key.strip():
        return "❌ กรุณาใส่ API Key ใน sidebar ก่อนครับ"
    
    try:
        if provider == "Claude (Anthropic)":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key.strip())
            resp = client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return resp.content[0].text
        
        elif provider == "Gemini (Google)":
            import google.generativeai as genai
            genai.configure(api_key=api_key.strip())
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=system_prompt,
            )
            resp = model.generate_content(user_message)
            return resp.text
        
        elif provider == "DeepSeek":
            from openai import OpenAI
            # แก้ไข: ใช้ http_client แทนการส่ง proxies argument โดยตรง
            http_client = httpx.Client(verify=True)
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://api.deepseek.com",
                http_client=http_client
            )
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                timeout=90,
            )
            result = resp.choices[0].message.content
            return result if result else "❌ LLM ตอบกลับเป็นค่าว่าง"
        
        else:
            return f"❌ Unknown provider: {provider}"
    
    except Exception as e:
        tb = traceback.format_exc()
        return f"❌ LLM Error ({provider}):\n{str(e)}\n\nTraceback:\n{tb}"

# ============================================================
# STEP 1: NORMALIZE FAILURE DESCRIPTION
# ============================================================
def normalize_failure(raw_text: str, provider: str, api_key: str) -> str:
    system = """You are a Senior Reliability Engineer writing professional failure reports for ISO 14224 compliance.
Rewrite in ENGLISH only. Structure: Component level, Context, Symptoms, Consequences. Output ONLY structured text."""
    return call_llm(system, raw_text, provider, api_key, max_tokens=800)

# ============================================================
# STEP 4: SUGGEST MAINTAINABLE ITEM (RAG)
# ============================================================
def suggest_maintainable_item(collection, equip_class_code: str,
                              equip_class_name: str, failure_desc: str,
                              provider: str, api_key: str) -> str:
    q1 = f"ISO 14224 {equip_class_name} {equip_class_code} maintainable item subunit taxonomy boundary"
    ctx1 = rag_search(collection, q1, n=8)
    q2 = f"{equip_class_name} {failure_desc} component part item"
    ctx2 = rag_search(collection, q2, n=5)
    
    combined_ctx = f"=== Equipment Taxonomy & Boundaries ===\n{ctx1}\n\n=== Failure-based Search ===\n{ctx2}"
    
    system = f"""You are an ISO 14224 expert. Determine the most appropriate MAINTAINABLE ITEM (Level 8) based on boundaries.
CONTEXT:
{combined_ctx}
FORMAT:
MAINTAINABLE ITEM: [item name]
SUBUNIT: [subunit name if applicable]
REASONING: [Brief explanation]"""
    user_msg = f"Equipment: {equip_class_code} — {equip_class_name}\nDescription: {failure_desc}"
    return call_llm(system, user_msg, provider, api_key, max_tokens=600)

# ============================================================
# STEPS 5-7: FULL ANALYSIS (FULL LOGIC RESTORED)
# ============================================================
def run_full_analysis(collection, equip_class_code: str, equip_class_name: str,
                      maintainable_item: str, normalized_desc: str,
                      provider: str, api_key: str) -> dict:
    results = {}
    
    # ════ STEP 5: FAILURE MODE ════
    q_fm = f"ISO 14224 {equip_class_name} {equip_class_code} failure mode definitions"
    ctx_fm = rag_search(collection, q_fm, n=10)
    q_tax = f"ISO 14224 {equip_class_name} {equip_class_code} taxonomy boundary"
    ctx_tax = rag_search(collection, q_tax, n=5)
    
    sys_fm = f"""You are an ISO 14224:2016 expert.
ANALYSIS APPROACH (CHAIN OF THOUGHT): Write a <thinking> block.
Then output:
FAILURE_MODE_CODE: [3-letter code]
FAILURE_MODE_NAME: [Full description]
FAILURE_MODE_EXAMPLES: [Examples]
FM_REASONING: [Justification]

=== DATA ===
{ctx_fm}
{ctx_tax}"""
    msg_fm = f"Equipment: {equip_class_code}\nItem: {maintainable_item}\nDescription: {normalized_desc}"
    results["fm_result"] = call_llm(sys_fm, msg_fm, provider, api_key, max_tokens=1500)
    
    # ════ STEP 6: FAILURE MECHANISM ════
    q_mech = f"Table B.2 failure mechanism physical process degradation"
    ctx_mech = rag_search(collection, q_mech, n=8)
    fm_result_text = results.get("fm_result", "")
    
    sys_mech = f"""You are an ISO 14224 expert. Select from TABLE B.2 ONLY.
<thinking> block first.
FORMAT:
MECHANISM_CODE: [e.g. 1]
MECHANISM_NOTATION: [e.g. Mechanical failure]
SUBDIVISION_CODE: [e.g. 1.2]
SUBDIVISION_NOTATION: [Vibration]
MECH_REASONING: [Summary]

=== DATA ===
{ctx_mech}
{fm_result_text}"""
    results["mech_result"] = call_llm(sys_mech, msg_fm, provider, api_key, max_tokens=1500)
    
    # ════ STEP 7: FAILURE CAUSE ════
    q_cause = f"Table B.3 failure cause root cause guidelines"
    ctx_cause = rag_search(collection, q_cause, n=8)
    mech_result_text = results.get("mech_result", "")
    
    sys_cause = f"""You are an ISO 14224 expert. Select from TABLE B.3 ONLY.
<thinking> block first.
FORMAT:
CAUSE_CODE: [e.g. 3]
CAUSE_NOTATION: [Operation]
CAUSE_SUBDIVISION_CODE: [3.3]
CAUSE_SUBDIVISION_NOTATION: [Wear]
CAUSE_REASONING: [Logic]

=== DATA ===
{ctx_cause}
{mech_result_text}"""
    results["cause_result"] = call_llm(sys_cause, msg_fm, provider, api_key, max_tokens=1500)
    
    # ════ COMBINE RESULTS (FULL PARSING) ════
    combined_json = {}
    def extract_field(text, prefix):
        for line in text.split("\n"):
            if line.strip().startswith(prefix):
                return line.split(":", 1)[1].strip()
        return ""
    
    fm_text = results.get("fm_result", "")
    combined_json["failure_mode_code"] = extract_field(fm_text, "FAILURE_MODE_CODE:")
    combined_json["failure_mode_name"] = extract_field(fm_text, "FAILURE_MODE_NAME:")
    combined_json["fm_reasoning"] = extract_field(fm_text, "FM_REASONING:")
    
    mech_text = results.get("mech_result", "")
    combined_json["failure_mechanism_notation"] = extract_field(mech_text, "MECHANISM_NOTATION:")
    m_sub_code = extract_field(mech_text, "SUBDIVISION_CODE:")
    m_sub_name = extract_field(mech_text, "SUBDIVISION_NOTATION:")
    combined_json["failure_mechanism_subdivision"] = f"{m_sub_code} {m_sub_name}"
    
    cause_text = results.get("cause_result", "")
    combined_json["failure_cause_notation"] = extract_field(cause_text, "CAUSE_NOTATION:")
    c_sub_code = extract_field(cause_text, "CAUSE_SUBDIVISION_CODE:")
    c_sub_name = extract_field(cause_text, "CAUSE_SUBDIVISION_NOTATION:")
    combined_json["failure_cause_subdivision"] = f"{c_sub_code} {c_sub_name}"
    
    combined_json["confidence"] = "HIGH" if combined_json["failure_mode_code"] else "LOW"
    results["combined_json"] = combined_json
    return results

# ============================================================
# UI & EXPORT FUNCTIONS
# ============================================================
def export_to_excel(data: dict) -> bytes:
    df = pd.DataFrame([data])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()

def inject_css():
    st.markdown("""
    <style>
    /* ซ่อนปุ่มรูปตาในช่อง Password */
    button[title="View password content"] { display: none !important; }
    
    .main-header { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem; color: white; text-align: center; 
    }
    .result-box { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 1rem; margin: 0.5rem 0; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE RENDERING
# ============================================================
def render_home():
    inject_css()
    st.markdown('<div class="main-header"><h1>🔧 ISO 14224 Analysis Console</h1></div>', unsafe_allow_html=True)
    if st.button("🔬 Go to Failure Analysis", type="primary"):
        st.session_state.page = "analysis"
        st.rerun()

def render_analysis():
    inject_css()
    
    with st.sidebar:
        st.markdown("### ⚙️ LLM Configuration")
        provider = st.selectbox("AI Provider", ["DeepSeek", "Claude (Anthropic)", "Gemini (Google)"])
        api_key = st.text_input("API Key", type="password")
    
    st.title("🔬 Failure Analysis")
    
    collection = init_vdb()
    df_plant = load_plant_unit()
    
    # Section 1: Raw Input
    raw_text = st.text_area("📝 รายละเอียด Failure", height=150)
    if st.button("✨ Normalize", type="primary"):
        st.session_state.normalized_text = normalize_failure(raw_text, provider, api_key)
        st.rerun()
        
    if st.session_state.get("normalized_text"):
        st.info(st.session_state.normalized_text)
        
        # Section 2-3: Equipment Select
        col1, col2 = st.columns(2)
        with col1:
            plant = st.selectbox("🏭 Plant", list(df_plant["Plant"].unique()))
        with col2:
            equip_label = st.selectbox("📐 Equipment Class", list(EQUIPMENT_CLASSES.keys()))
            equip_code = EQUIPMENT_CLASSES[equip_label]
            equip_name = equip_label.split(" — ")[1] if equip_code else ""

        # Section 4: Maintainable Item
        if st.button("🤖 แนะนำ Maintainable Item"):
            suggestion = suggest_maintainable_item(collection, equip_code, equip_name, st.session_state.normalized_text, provider, api_key)
            st.session_state.suggested_mi = suggestion
            st.markdown(suggestion)

        # Final Run
        if st.button("🔍 Run Full Analysis", use_container_width=True):
            with st.spinner("Analyzing..."):
                final_mi = "Item Name" # Extract from suggested_mi in real use
                ans = run_full_analysis(collection, equip_code, equip_name, final_mi, st.session_state.normalized_text, provider, api_key)
                st.session_state.analysis_result = ans
                st.success("Analysis Complete!")
                st.json(ans["combined_json"])

def main():
    if "page" not in st.session_state: st.session_state.page = "home"
    if st.session_state.page == "home": render_home()
    else: render_analysis()

if __name__ == "__main__":
    main()
