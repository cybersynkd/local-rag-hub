import os
import json
import subprocess
import datetime
import numpy as np
import streamlit as st
from google import genai

VECTOR_FILE = "local_vector_index.json"
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")
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

st.sidebar.markdown("**Active Tools**")
enable_rag = st.sidebar.checkbox("Semantic RAG Embeddings", value=True)
enable_exec = st.sidebar.checkbox("Python Script Execution Tool", value=True)

if enable_exec:
    st.subheader("⚡ Python Automation Terminal")
    script_input = st.text_area("Enter Python code to execute locally:", "print('Sovereign Node Active')")
    if st.button("Run Script"):
        output = execute_python_script(script_input)
        st.code(output)
        save_new_entry_with_embedding(f"Executed Script: {script_input} | Output: {output}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
