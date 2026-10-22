import os
import json
import subprocess
import datetime
import time
import numpy as np
import streamlit as st
from google import genai

VECTOR_FILE = "local_vector_index.json"
CACHE_FILE = "local_response_cache.json"

st.set_page_config(
    page_title="Sovereign Personal AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 1. PERSISTENT RLT STATE ENGINE ───────────────────────────
class RecurrentLoopStateTracker:
    """Engine for continuous state propagation preserved across Streamlit rerun cycles."""
    def __init__(self, layer_depth=48):
        self.layer_depth = layer_depth
        self.hidden_state = 0.5
        self.swa_cache = []

    def step(self, token_input):
        active_computation = self._fuse_state(token_input, self.hidden_state, self.swa_cache)
        self.hidden_state = active_computation['next_hidden']
        self.swa_cache = active_computation['updated_swa']
        return active_computation['output']

    def _fuse_state(self, token, h_prev, cache):
        token_str = str(token)
        next_h = (hash(token_str + str(h_prev)) % 10000) / 10000.0
        updated_cache = (cache + [token_str])[-self.layer_depth:]
        depth_traversal = len(updated_cache) * self.layer_depth
        return {
            'next_hidden': next_h,
            'updated_swa': updated_cache,
            'output': f"RLT Vector [Depth: {depth_traversal} | State: {next_h:.4f}]"
        }

# Cache resource ensures RLT state persists through app updates without resetting
@st.cache_resource
def get_rlt_engine():
    return RecurrentLoopStateTracker(layer_depth=48)

rlt_engine = get_rlt_engine()

# ── 2. SIDEBAR & TOOL CONFIGURATION ──────────────────────────
with st.sidebar:
    st.title("🌱 Control Panel")
    
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key", type="password")

    st.markdown("### Active Model & Settings")
    selected_model = st.selectbox(
        "Free Tier Model",
        ["gemini-3.6-flash", "gemini-3.1-flash-lite"],
        index=0,
        help="Flash models maximize free tier allowances and support high context limits."
    )
    
    enable_rag = st.checkbox("Semantic RAG Embeddings", value=True)
    enable_cache = st.checkbox("Aggressive Local Caching", value=True, help="Saves quota by caching repeated prompts.")
    enable_exec = st.checkbox("Python Script Execution Tool", value=True)

    st.markdown("---")
    st.markdown("### RLT Lattice Telemetry")
    rlt_metric_col1, rlt_metric_col2 = st.columns(2)
    with rlt_metric_col1:
        st.metric("Hidden State", f"{rlt_engine.hidden_state:.4f}")
    with rlt_metric_col2:
        st.metric("SWA Depth", len(rlt_engine.swa_cache))

client = genai.Client(api_key=api_key) if api_key else None

# ── 3. DATA & CACHING PIPELINES ──────────────────────────────
def get_embedding(text):
    if not client:
        return None
    for attempt in range(3):
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return response.embedding.values
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return None
    return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_query_vector_store(query_text, top_k=3):
    if not os.path.exists(VECTOR_FILE):
        return []
    try:
        with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
    except Exception:
        return []
        
    query_emb = get_embedding(query_text)
    if not query_emb:
        query_terms = set(query_text.lower().split())
        scored = []
        for r in records:
            text = r.get('text', '').lower()
            matches = sum(1 for term in query_terms if term in text)
            if matches > 0:
                scored.append((matches, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]
        
    scored = []
    for r in records:
        text_emb = r.get('embedding')
        if text_emb:
            score = cosine_similarity(query_emb, text_emb)
            scored.append((score, r))
        else:
            scored.append((0.0, r))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

def save_new_entry_with_embedding(text, timestamp):
    records = []
    if os.path.exists(VECTOR_FILE):
        try:
            with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception:
            records = []
    
    if any(r.get('text') == text for r in records):
        return

    emb = get_embedding(text)
    records.append({
        "time": timestamp,
        "text": text,
        "embedding": emb if emb else []
    })
    with open(VECTOR_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)

def check_cache(prompt):
    if not enable_cache or not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            return cache.get(prompt)
    except Exception:
        return None

def save_cache(prompt, response):
    if not enable_cache:
        return
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    cache[prompt] = response
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def execute_python_script(script_code):
    try:
        result = subprocess.run(
            ["python3", "-c", script_code],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return f"Execution Success:\n{result.stdout}"
        else:
            return f"Execution Error:\n{result.stderr}"
    except Exception as e:
        return f"Execution Failed: {str(e)}"

def synthesize_response(prompt, context_snippets):
    if not client:
        return "Logged to memory, but synthesis failed: Missing Gemini API Key."
    
    cached = check_cache(prompt)
    if cached:
        return f"{cached}\n\n*(Retrieved from local cache)*"

    # Advance continuous state vector
    rlt_sig = rlt_engine.step(prompt)

    full_prompt = f"""You are a personal intelligence assistant operating within a Recurrent Looped Transformer (RLT) architectural framework. 
Current Recurrent Lattice Vector: {rlt_sig}

Use the following retrieved historical context and ongoing continuous state to answer the user's prompt accurately.

Retrieved Context:
{context_snippets}

User Prompt: {prompt}
"""
    
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=full_prompt,
            )
            text_resp = response.text
            save_cache(prompt, text_resp)
            return text_resp
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < 3:
                    time.sleep(2 ** attempt + 1)
                    continue
            return f"Synthesis failed due to rate limit or error: {error_str}"
    return "Synthesis failed: Max retries exceeded for free tier limits."

# ── 4. MAIN INTERFACE ─────────────────────────────────────────
st.title("🌱 Sovereign Personal Intelligence Hub")

if enable_exec:
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚡ Quick Python Runner")
        quick_script = st.text_area("Snippet:", "print('Node Active')")
        if st.button("Run in Background", use_container_width=True):
            output = execute_python_script(quick_script)
            st.code(output, language="python")
            
            rlt_exec_sig = rlt_engine.step(f"ScriptExec: {quick_script}")
            save_new_entry_with_embedding(
                f"Executed Script: {quick_script} | Output: {output} | {rlt_exec_sig}", 
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            )
            st.rerun()

# Chat Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or log an action..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    if enable_rag:
        with st.status("Querying Local Vector Store...", expanded=False):
            relevant = semantic_query_vector_store(prompt, top_k=3)
            context_snippets = "\n".join([f"- [{r.get('time', 'Unknown')}] {r['text']}" for r in relevant]) if relevant else "No prior context found."
    else:
        context_snippets = "RAG disabled."

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing response via RLT pipeline..."):
            response_content = synthesize_response(prompt, context_snippets)
            st.markdown(response_content)
            
    save_new_entry_with_embedding(prompt, timestamp)
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.rerun()
