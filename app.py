import os
import json
import subprocess
import datetime
import numpy as np
import streamlit as st
from google import genai

VECTOR_FILE = "local_vector_index.json"

st.set_page_config(
    page_title="Sovereign Personal AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Key Setup
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

client = genai.Client(api_key=api_key) if api_key else None

def get_embedding(text):
    if not client:
        return None
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return response.embedding.values
    except Exception:
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
    try:
        full_prompt = f"""You are a personal intelligence assistant. Use the following retrieved historical context to answer the user's prompt accurately.

Retrieved Context:
{context_snippets}

User Prompt: {prompt}
"""
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
        )
        return response.text
    except Exception as e:
        return f"Synthesis failed: {str(e)}"

# Unified Main Screen Layout
st.title("🌱 Sovereign Personal Intelligence Hub")

# Sidebar Controls & Terminal
st.sidebar.markdown("**Active Tools**")
enable_rag = st.sidebar.checkbox("Semantic RAG Embeddings", value=True)
enable_exec = st.sidebar.checkbox("Python Script Execution Tool", value=True)

if enable_exec:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Python Runner")
    quick_script = st.sidebar.text_area("Snippet:", "print('Node Active')")
    if st.sidebar.button("Run in Background"):
        output = execute_python_script(quick_script)
        st.sidebar.code(output)
        save_new_entry_with_embedding(f"Executed Script: {quick_script} | Output: {output}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

# Main Chat Interface Loop
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
        relevant = semantic_query_vector_store(prompt, top_k=3)
        context_snippets = "\n".join([f"- [{r.get('time', 'Unknown')}] {r['text']}" for r in relevant]) if relevant else "No prior context found."
    else:
        context_snippets = "RAG disabled."

    response_content = synthesize_response(prompt, context_snippets)
    save_new_entry_with_embedding(prompt, timestamp)

    with st.chat_message("assistant"):
        st.markdown(response_content)
    st.session_state.messages.append({"role": "assistant", "content": response_content})
