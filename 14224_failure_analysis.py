#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14224_failure_analysis.py
=========================
ISO 14224 Failure Analysis — Streamlit App (Optimized with CoT & Enhanced RAG)
ระบบวิเคราะห์ Failure ตามมาตรฐาน ISO 14224:2016
Version: 4.0 - Fixed for Render.com Deployment
"""
import streamlit as st
import pandas as pd
import json
import io
import os
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
# EQUIPMENT CLASS MASTER LIST (ISO 14224)
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

@st.cache_resource          # ✅ มีอยู่แล้วที่นี่
def init_vdb():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path=CONFIG["chroma_path"])
        collection = client.get_collection(
            name=CONFIG["collection_name"],
            embedding_function=ef,
        )
        return collection
    except Exception as e:
        st.error(f"❌ VDB Error: {e}")
        return None
# ============================================================
# RAG SEARCH
# ============================================================
def rag_search(collection, query: str, n: int = 10, topic_filter: str = None) -> str:
    if collection is None:
        return "[VDB not available]"
    
    kwargs = {"query_texts": [query], "n_results": n}
    if topic_filter:
        kwargs["where"] = {"topic": {"$eq": topic_filter}}
    
    try:
        results = collection.query(**kwargs)
    except Exception:
        results = collection.query(query_texts=[query], n_results=n)
    
    if not results["ids"][0]:
        results = collection.query(query_texts=[query], n_results=n)
    
    contexts = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        title = meta.get("title", "")
        page = meta.get("page_number", "?")
        contexts.append(f"[Source {i+1} | Page {page}] {title}\n{doc}")
    
    return "\n---\n".join(contexts) if contexts else "[No relevant data found]"

# ============================================================
# LLM WRAPPER
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
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://api.deepseek.com",
            )
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                timeout=60,
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

Your task is to rewrite the user's failure description into clear, structured, professional English optimized for Root Cause Analysis (RCA).

RULES:
1. Rewrite in ENGLISH only.
2. Structure:
   - What failed (Component level)
   - Operational Context (Explicitly extract temperature, pressure, speed, load, or environmental conditions if mentioned)
   - How it was detected & Symptoms observed
   - Consequences (e.g., shutdown, leak)
3. Be concise but complete — preserve ALL technical details.
4. Output ONLY the structured text, no conversational fillers."""
    
    return call_llm(system, raw_text, provider, api_key, max_tokens=800)

# ============================================================
# STEP 4: SUGGEST MAINTAINABLE ITEM (RAG)
# ============================================================
def suggest_maintainable_item(collection, equip_class_code: str,
                              equip_class_name: str, failure_desc: str,
                              provider: str, api_key: str) -> str:
    q1 = f"ISO 14224 {equip_class_name} {equip_class_code} maintainable item subunit taxonomy boundary diagram definition"
    ctx1 = rag_search(collection, q1, n=8)
    
    q2 = f"{equip_class_name} {failure_desc} component part item"
    ctx2 = rag_search(collection, q2, n=5)
    
    combined_ctx = f"=== Equipment Taxonomy & Boundaries ===\n{ctx1}\n\n=== Failure-based Search ===\n{ctx2}"
    
    system = f"""You are an ISO 14224 expert. Determine the most appropriate MAINTAINABLE ITEM (Level 8) based on the equipment boundaries.

CONTEXT FROM ISO 14224 DATABASE:
{combined_ctx}

RULES:
1. Review the equipment boundary to ensure the component belongs to "{equip_class_code} ({equip_class_name})" and not an adjacent system.
2. If exact match isn't found, find the BEST MATCH from similar classes.
3. Return your answer EXACTLY in this format:
MAINTAINABLE ITEM: [item name]
SUBUNIT: [subunit name if applicable]
REASONING: [Brief explanation verifying it falls within the equipment boundary]"""
    
    user_msg = f"Equipment: {equip_class_code} — {equip_class_name}\nDescription: {failure_desc}"
    
    return call_llm(system, user_msg, provider, api_key, max_tokens=600)

# ============================================================
# STEPS 5-7: FULL ANALYSIS (RAG + LLM with CoT)
# ============================================================
def run_full_analysis(collection, equip_class_code: str, equip_class_name: str,
                      maintainable_item: str, normalized_desc: str,
                      provider: str, api_key: str) -> dict:
    results = {}
    
    # ════ STEP 5: FAILURE MODE ════
    q_fm = f"ISO 14224 {equip_class_name} {equip_class_code} failure mode definitions descriptions symptoms"
    ctx_fm = rag_search(collection, q_fm, n=10)
    
    q_tax = f"ISO 14224 {equip_class_name} {equip_class_code} taxonomy subunit maintainable item boundary"
    ctx_tax = rag_search(collection, q_tax, n=5)
    
    short_desc = normalized_desc[:120]
    q_evidence = f"{equip_class_name} {maintainable_item} {short_desc} symptom"
    ctx_evidence = rag_search(collection, q_evidence, n=5)
    
    sys_fm = f"""You are an ISO 14224:2016 expert.

=== FAILURE MODE MATRIX & DEFINITIONS ===
{ctx_fm}

=== EQUIPMENT TAXONOMY ===
{ctx_tax}

=== SUPPORTING EVIDENCE ===
{ctx_evidence}

TASK: Find the BEST MATCHING failure mode for equipment class "{equip_class_code}".

ANALYSIS APPROACH (CHAIN OF THOUGHT):
Before formatting your final output, write a `<thinking>` block where you:
1. Confirm the maintainable item is within the equipment boundary.
2. Compare the observed symptoms directly against the official definitions.
3. Eliminate modes that do not fit the symptoms.

After the `<thinking>` block, output EXACTLY this format:
FAILURE_MODE_CODE: [3-letter code]
FAILURE_MODE_NAME: [Full description]
FAILURE_MODE_EXAMPLES: [Examples from the ISO table]
FM_REASONING: [Your summarized justification]"""
    
    msg_fm = f"Equipment: {equip_class_code} — {equip_class_name}\nItem: {maintainable_item}\nDescription: {normalized_desc}"
    results["fm_result"] = call_llm(sys_fm, msg_fm, provider, api_key, max_tokens=1500)
    
    # ════ STEP 6: FAILURE MECHANISM ════
    q_mech = f"Table B.2 failure mechanism physical process degradation definitions"
    ctx_mech = rag_search(collection, q_mech, n=8)
    
    q_mech_ev = f"failure mechanism {maintainable_item} {short_desc} physical process"
    ctx_mech_ev = rag_search(collection, q_mech_ev, n=5)
    
    fm_result_text = results.get("fm_result", "")
    
    sys_mech = f"""You are an ISO 14224:2016 expert. Select from TABLE B.2 (Failure Mechanism) ONLY.

=== TABLE B.2 DEFINITIONS ===
{ctx_mech}

=== SUPPORTING EVIDENCE ===
{ctx_mech_ev}

=== IDENTIFIED FAILURE MODE ===
{fm_result_text}

ANALYSIS APPROACH (CHAIN OF THOUGHT):
Write a `<thinking>` block where you:
1. Identify how the Failure Mode physicalized (HOW it broke).
2. Ensure Engineering Consistency.

After the `<thinking>` block, output EXACTLY:
MECHANISM_CODE: [e.g. 1]
MECHANISM_NOTATION: [e.g. Mechanical failure]
SUBDIVISION_CODE: [e.g. 1.2]
SUBDIVISION_NOTATION: [e.g. Vibration]
MECH_REASONING: [Brief summary of why]"""
    
    msg_mech = f"Equipment: {equip_class_code}\nItem: {maintainable_item}\nDescription: {normalized_desc}"
    results["mech_result"] = call_llm(sys_mech, msg_mech, provider, api_key, max_tokens=1500)
    
    # ════ STEP 7: FAILURE CAUSE ════
    q_cause = f"Table B.3 failure cause definitions root cause guidelines"
    ctx_cause = rag_search(collection, q_cause, n=8)
    
    q_cause_ev = f"failure root cause {maintainable_item} {short_desc} why reason"
    ctx_cause_ev = rag_search(collection, q_cause_ev, n=5)
    
    mech_result_text = results.get("mech_result", "")
    
    sys_cause = f"""You are an ISO 14224:2016 expert. Select from TABLE B.3 (Failure Causes) ONLY.

=== TABLE B.3 DATA ===
{ctx_cause}

=== SUPPORTING EVIDENCE ===
{ctx_cause_ev}

=== PREVIOUS ANALYSIS (FM & MECH) ===
{fm_result_text[:500]}
{mech_result_text[:500]}

ANALYSIS APPROACH (CHAIN OF THOUGHT):
Write a `<thinking>` block where you:
1. Connect the Root Cause (WHY) -> Mechanism (HOW) -> Mode (WHAT).
2. Identify the PRIMARY cause.

After the `<thinking>` block, output EXACTLY:
CAUSE_CODE: [e.g. 3]
CAUSE_NOTATION: [e.g. Operation/maintenance]
CAUSE_SUBDIVISION_CODE: [e.g. 3.3]
CAUSE_SUBDIVISION_NOTATION: [e.g. Wear and tear]
CAUSE_REASONING: [Brief summary of the root cause logic]"""
    
    msg_cause = f"Equipment: {equip_class_code}\nItem: {maintainable_item}\nDescription: {normalized_desc}"
    results["cause_result"] = call_llm(sys_cause, msg_cause, provider, api_key, max_tokens=1500)
    
    # ════ COMBINE RESULTS ════
    combined_json = {}
    
    def extract_field(text, prefix):
        for line in text.split("\n"):
            if line.strip().startswith(prefix):
                return line.split(":", 1)[1].strip()
        return ""
    
    # Parse FM
    fm_text = results.get("fm_result", "")
    combined_json["failure_mode_code"] = extract_field(fm_text, "FAILURE_MODE_CODE:")
    combined_json["failure_mode_name"] = extract_field(fm_text, "FAILURE_MODE_NAME:")
    combined_json["failure_mode_examples"] = extract_field(fm_text, "FAILURE_MODE_EXAMPLES:")
    combined_json["fm_reasoning"] = extract_field(fm_text, "FM_REASONING:")
    
    # Parse Mechanism
    mech_text = results.get("mech_result", "")
    combined_json["failure_mechanism_notation"] = extract_field(mech_text, "MECHANISM_NOTATION:")
    combined_json["mechanism_subdivision_code"] = extract_field(mech_text, "SUBDIVISION_CODE:")
    sub_notation = extract_field(mech_text, "SUBDIVISION_NOTATION:")
    combined_json["failure_mechanism_subdivision"] = f"{combined_json.get('mechanism_subdivision_code','')} {sub_notation}".strip()
    combined_json["mech_reasoning"] = extract_field(mech_text, "MECH_REASONING:")
    
    # Parse Cause
    cause_text = results.get("cause_result", "")
    combined_json["failure_cause_notation"] = extract_field(cause_text, "CAUSE_NOTATION:")
    combined_json["cause_subdivision_code"] = extract_field(cause_text, "CAUSE_SUBDIVISION_CODE:")
    cause_sub = extract_field(cause_text, "CAUSE_SUBDIVISION_NOTATION:")
    combined_json["failure_cause_subdivision"] = f"{combined_json.get('cause_subdivision_code','')} {cause_sub}".strip()
    combined_json["cause_reasoning"] = extract_field(cause_text, "CAUSE_REASONING:")
    
    # Validate Confidence
    filled_fields = sum(1 for v in combined_json.values() if v and v != "N/A" and v.strip() != "")
    combined_json["confidence"] = "HIGH" if filled_fields >= 10 else "MEDIUM"
    
    results["combined_json"] = combined_json
    return results

# ============================================================
# EXPORT TO EXCEL
# ============================================================
def export_to_excel(data: dict) -> bytes:
    df = pd.DataFrame([data])
    col_order = [
        "analysis_date", "plant", "unit", "machine_type",
        "equipment_class_code", "equipment_class_name",
        "original_description", "normalized_description",
        "maintainable_item",
        "failure_mode_code", "failure_mode_name", "failure_mode_examples", "fm_reasoning",
        "failure_mechanism_notation", "failure_mechanism_subdivision", "mech_reasoning",
        "failure_cause_notation", "failure_cause_subdivision", "cause_reasoning",
        "confidence", "llm_provider",
    ]
    existing_cols = [c for c in col_order if c in df.columns]
    remaining = [c for c in df.columns if c not in col_order]
    df = df[existing_cols + remaining]
    
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Failure Analysis")
        ws = writer.sheets["Failure Analysis"]
        for col_cells in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col_cells)
            col_letter = col_cells[0].column_letter
            ws.column_dimensions[col_letter].width = min(max_len + 4, 50)
    
    return buf.getvalue()

# ============================================================
# CUSTOM CSS
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    .main-header { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
        padding: 2rem 3rem; 
        border-radius: 16px; 
        margin-bottom: 2rem; 
        color: white; 
        text-align: center; 
    }
    .main-header h1 { color: #e2e8f0; margin: 0; font-size: 2rem; }
    .main-header p { color: #94a3b8; margin: 0.5rem 0 0; }
    .result-box { 
        background: #f0fdf4; 
        border-left: 4px solid #22c55e; 
        padding: 1rem 1.5rem; 
        border-radius: 0 8px 8px 0; 
        margin: 0.5rem 0; 
    }
    div.stButton > button[kind="primary"] { 
        width: 100%; 
        padding: 1rem 2rem; 
        font-size: 1.2rem; 
    }
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# HOME PAGE
# ============================================================
def render_home():
    inject_css()
    st.markdown("""
    <div class="main-header">
    <h1>🔧 ISO 14224 Reliability Console (Optimized)</h1>
    <p>Bad Actor Analysis System — Guided by CoT & RAG Enhancement</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### เลือกเครื่องมือที่ต้องการ")
        st.write("")
        if st.button("🔬 **Failure Analysis** — วิเคราะห์ความเสียหายตาม ISO 14224", 
                     type="primary", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()
    
    st.divider()

# ============================================================
# ANALYSIS PAGE
# ============================================================
def render_analysis():
    inject_css()
    
    col_back, col_title = st.columns([1, 11])
    with col_back:
        if st.button("← กลับ"):
            st.session_state.page = "home"
            st.rerun()
        
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col_title:
        st.markdown("## 🔬 Failure Analysis — ISO 14224")
    
    # Sidebar ✅ แก้ไข Indentation แล้ว
    with st.sidebar:
        st.markdown("### ⚙️ LLM Configuration")
        provider = st.selectbox("AI Provider", ["Claude (Anthropic)", "Gemini (Google)", "DeepSeek"])
        
        # ดึงจาก Environment Variables (สำหรับ Render.com)
        if provider == "Claude (Anthropic)":
            default_key = os.environ.get("ANTHROPIC_API_KEY", "")
            api_key = st.text_input("Anthropic API Key", value=default_key, type="password")
        elif provider == "Gemini (Google)":
            default_key = os.environ.get("GEMINI_API_KEY", "")
            api_key = st.text_input("Gemini API Key", value=default_key, type="password")
        else:  # DeepSeek
            default_key = os.environ.get("DEEPSEEK_API_KEY", "")
            api_key = st.text_input("DeepSeek API Key", value=default_key, type="password")
        
        # Test Connection Button
        if api_key and st.button("🧪 ทดสอบการเชื่อมต่อ API"):
            with st.spinner("กำลังทดสอบ..."):
                test_result = call_llm(
                    "You are a helpful assistant.",
                    "Say 'OK' if you can read this.",
                    provider, api_key, max_tokens=10
                )
                if test_result and not test_result.startswith("❌"):
                    st.success(f"✅ API ทำงานปกติ: {test_result}")
                else:
                    st.error(f"❌ API Error: {test_result}")
    
    # Initialize
    collection = init_vdb()
    df_plant = load_plant_unit()
    
    # SECTION 1
    st.markdown("### ① Failure Capture — บันทึกรายละเอียดความเสียหาย")
    
    col_raw, col_norm = st.columns(2)
    
    with col_raw:
        failure_raw = st.text_area("📝 รายละเอียด Failure", 
                                   height=200,
                                   key="failure_raw_input",
                                   help="กรอกรายละเอียดความเสียหายที่นี่")
    
    with col_norm:
        st.markdown("**📋 Normalized Description**")
        if st.session_state.get("normalized_text", ""):
            st.text_area("Normalized Description", 
                        value=st.session_state.normalized_text, 
                        height=200, 
                        disabled=True, 
                        key="norm_display")
        else:
            st.info("ผลลัพธ์จะแสดงที่นี่หลังจากกด Normalize")
    
    # Normalize Button
    can_normalize = bool(failure_raw and failure_raw.strip() and api_key and api_key.strip())
    
    if st.button("✨ Normalize", type="primary", disabled=not can_normalize):
        with st.spinner("🔄 กำลังเรียบเรียงและสกัด Operational Context..."):
            try:
                res = normalize_failure(failure_raw, provider, api_key)
                
                if res.startswith("❌"):
                    st.session_state.normalize_error = res
                    st.session_state.normalized_text = ""
                    st.error(res)
                elif res and len(res.strip()) > 0:
                    st.session_state.normalized_text = res
                    st.session_state.normalize_error = ""
                    st.success("✅ Normalize สำเร็จ!")
                    st.rerun()
                else:
                    st.error("❌ ไม่ได้รับ response จาก LLM หรือ response ว่างเปล่า")
                    
            except Exception as e:
                error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
                st.error(error_msg)
                with st.expander("📋 Traceback"):
                    st.code(traceback.format_exc())
    
    st.divider()
    
    # SECTION 2 & 3
    st.markdown("### ② & ③ Equipment Details")
    
    col_p, col_u, col_m, col_e = st.columns(4)
    
    with col_p:
        selected_plant = st.selectbox("🏭 Plant", 
                                     ["-- เลือก --"] + list(df_plant["Plant"].dropna().unique()),
                                     key="plant_select")
    
    with col_u:
        units = df_plant[df_plant["Plant"] == selected_plant]["Unit"].dropna().unique() \
                if selected_plant != "-- เลือก --" else []
        selected_unit = st.selectbox("🔧 Unit", 
                                    ["-- เลือก --"] + list(units),
                                    key="unit_select")
    
    with col_m:
        machines = df_plant[(df_plant["Plant"] == selected_plant) & 
                           (df_plant["Unit"] == selected_unit)]["Machinetype"].dropna().unique() \
                  if selected_unit != "-- เลือก --" else []
        selected_machine = st.selectbox("⚙️ Machine Type", 
                                       ["-- เลือก --"] + list(machines),
                                       key="machine_select")
    
    with col_e:
        equip_options = list(EQUIPMENT_CLASSES.keys())
        selected_equip_label = st.selectbox("📐 Equipment Class", 
                                           equip_options,
                                           key="equip_select")
        equip_code = EQUIPMENT_CLASSES.get(selected_equip_label)
        equip_name = selected_equip_label.split(" — ")[1] if equip_code else ""
    
    st.divider()
    
    # SECTION 4
    st.markdown("### ④ Maintainable Item")
    
    col_mi, col_mi_btn = st.columns([3, 1])
    
    with col_mi:
        mi_input = st.text_input("🔩 Maintainable Item", 
                                value=st.session_state.get("suggested_mi", ""),
                                key="mi_input_field")
    
    with col_mi_btn:
        st.write("")
        st.write("")
        suggest_disabled = (not equip_code or not st.session_state.get("normalized_text", ""))
        if st.button("🤖 LLM แนะนำ", disabled=suggest_disabled):
            with st.spinner("🔍 กำลังตรวจสอบ Boundary & Taxonomy..."):
                try:
                    suggestion = suggest_maintainable_item(
                        collection, equip_code, equip_name, 
                        st.session_state.normalized_text, 
                        provider, api_key
                    )
                    st.session_state.suggested_mi_result = suggestion
                    
                    # Extract MAINTAINABLE ITEM
                    for line in suggestion.split("\n"):
                        if line.strip().upper().startswith("MAINTAINABLE ITEM:"):
                            st.session_state.suggested_mi = line.split(":", 1)[1].strip()
                            break
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.get("suggested_mi_result", ""):
        with st.expander("📖 รายละเอียดการแนะนำ", expanded=False):
            st.markdown(st.session_state.suggested_mi_result)
    
    st.divider()
    
    # SECTION 5-7
    st.markdown("### ⑤⑦ Failure Analysis (CoT Enabled)")
    
    final_mi = mi_input or st.session_state.get("suggested_mi", "")
    ready = (st.session_state.get("normalized_text", "") and 
             equip_code and 
             final_mi and
             selected_plant != "-- เลือก --" and
             selected_unit != "-- เลือก --")
    
    if st.button("🔍 **Run Full Analysis**", type="primary", 
                 use_container_width=True, disabled=not ready):
        
        with st.status("🔬 กำลังวิเคราะห์เชิงลึก (Chain of Thought)...", expanded=True) as status:
            try:
                analysis = run_full_analysis(
                    collection, equip_code, equip_name, final_mi,
                    st.session_state.normalized_text, provider, api_key
                )
                status.update(label="✅ วิเคราะห์เสร็จสิ้น!", state="complete")
                
                st.session_state.analysis_result = analysis
                st.session_state.analysis_json = analysis.get("combined_json", {})
                st.session_state.analysis_meta = {
                    "plant": selected_plant,
                    "unit": selected_unit,
                    "machine_type": selected_machine,
                    "equipment_class_code": equip_code,
                    "equipment_class_name": equip_name,
                    "maintainable_item": final_mi,
                }
            except Exception as e:
                status.update(label="❌ เกิดข้อผิดพลาด", state="error")
                st.error(f"❌ Analysis Error: {str(e)}")
                with st.expander("📋 Traceback"):
                    st.code(traceback.format_exc())
    
    # Display Results
    if st.session_state.get("analysis_result"):
        analysis = st.session_state.analysis_result
        
        st.markdown("---")
        st.markdown("### 🧠 กระบวนการคิดของ LLM (Chain of Thought)")
        
        tab_fm, tab_mech, tab_cause = st.tabs([
            "🎯 Mode Reasoning", 
            "⚙️ Mechanism Reasoning", 
            "🔍 Cause Reasoning"
        ])
        
        with tab_fm:
            st.markdown(analysis.get("fm_result", ""))
        with tab_mech:
            st.markdown(analysis.get("mech_result", ""))
        with tab_cause:
            st.markdown(analysis.get("cause_result", ""))
        
        # Final Output
        aj = st.session_state.get("analysis_json", {})
        if aj:
            st.markdown("### 📋 สรุปผลวิเคราะห์ (Final Output)")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.markdown("#### 🎯 Failure Mode")
                st.markdown(f"""
                <div class="result-box">
                    <strong style="font-size:1.3rem; color:#15803d;">
                        {aj.get('failure_mode_code', 'N/A')}
                    </strong><br>
                    <strong>{aj.get('failure_mode_name', 'N/A')}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r2:
                st.markdown("#### ⚙️ Mechanism")
                st.markdown(f"""
                <div class="result-box">
                    <strong style="font-size:1.1rem; color:#15803d;">
                        {aj.get('failure_mechanism_notation', 'N/A')}
                    </strong><br>
                    <span>→ {aj.get('failure_mechanism_subdivision', 'N/A')}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r3:
                st.markdown("#### 🔍 Root Cause")
                st.markdown(f"""
                <div class="result-box">
                    <strong style="font-size:1.1rem; color:#15803d;">
                        {aj.get('failure_cause_notation', 'N/A')}
                    </strong><br>
                    <span>→ {aj.get('failure_cause_subdivision', 'N/A')}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Export
            meta = st.session_state.get("analysis_meta", {})
            export_data = {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "plant": meta.get("plant", ""),
                "unit": meta.get("unit", ""),
                "machine_type": meta.get("machine_type", ""),
                "equipment_class_code": meta.get("equipment_class_code", ""),
                "equipment_class_name": meta.get("equipment_class_name", ""),
                "original_description": st.session_state.get("failure_raw_input", ""),
                "normalized_description": st.session_state.normalized_text,
                "maintainable_item": meta.get("maintainable_item", ""),
                **aj,
                "llm_provider": provider,
            }
            
            st.download_button(
                label="📥 ดาวน์โหลด Excel",
                data=export_to_excel(export_data),
                file_name=f"FailureAnalysis_{meta.get('equipment_class_code', 'XX')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary", 
                use_container_width=True
            )

# ============================================================
# MAIN
# ============================================================
def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "normalized_text" not in st.session_state:
        st.session_state.normalized_text = ""
    if "normalize_error" not in st.session_state:
        st.session_state.normalize_error = ""
    if "suggested_mi" not in st.session_state:
        st.session_state.suggested_mi = ""
    
    if st.session_state.page == "home":
        render_home()
    else:
        render_analysis()

if __name__ == "__main__":
    main()
