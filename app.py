import streamlit as st
import datetime
import os
import json
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

VECTOR_FILE = 'local_vector_index.json'

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

client = genai.Client(api_key=api_key) if api_key else None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

def get_embedding(text):
    return model.encode(text).tolist()

def cosine_similarity(v1, v2):
    a, b = np.array(v1), np.array(v2)
    if a.shape != b.shape:
        return 0.0
    dot_product = np.dot(a, b)
    norm1 = np.linalg.norm(a)
    norm2 = np.linalg.norm(b)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))

def save_vector_entry(text, timestamp):
    records = []
    if os.path.exists(VECTOR_FILE):
        try:
            with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception:
            records = []
    
    # Prevent duplicate text entries
    if any(r.get('text') == text for r in records):
        return

    records.append({
        "time": timestamp, 
        "text": text, 
        "vector": get_embedding(text)
    })
    with open(VECTOR_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)

# Automatically load and index local .txt files on startup if not already done
@st.cache_resource
def auto_ingest_local_files():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    local_files = glob.glob("*.txt")
    for file_path in local_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()
                
            paragraphs = [p.strip() for p in file_content.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [file_content]
                
            for i, chunk in enumerate(paragraphs):
                chunk_label = f"Local File [{file_path}] Part {i+1}/{len(paragraphs)}"
                save_vector_entry(f"{chunk_label}:\n{chunk}", timestamp)
        except Exception as e:
            print(f"Error auto-ingesting {file_path}: {e}")

auto_ingest_local_files()

def query_vector_store(query_text, top_k=3):
    if not os.path.exists(VECTOR_FILE):
        return []
    with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)
    q_vec = get_embedding(query_text)
    scored = []
    for r in records:
        vec = r.get('vector')
        if not vec or len(vec) != len(q_vec):
            continue
        score = cosine_similarity(q_vec, vec)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

def synthesize_response(prompt, context_snippets):
    if not client:
        return "Logged to memory, but synthesis failed: Missing Gemini API Key."
    try:
        full_prompt = f"""You are a personal intelligence assistant. Use the following retrieved historical context to answer the user's prompt accurately. If the context doesn't have the answer, rely on your general knowledge while keeping the user's records in mind.

Retrieved Context:
{context_snippets}

User Prompt: {prompt}
"""
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=full_prompt,
        )
        return response.text
    except Exception as e:
        return f"Logged to memory, but synthesis failed: {str(e)}"

st.title("🌱 Personal Intelligence Chat Hub")

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
    
    relevant = query_vector_store(prompt, top_k=3)
    
    if relevant:
        context_snippets = "\n".join([f"- [{r['time']}] {r['text']}" for r in relevant])
    else:
        context_snippets = "No prior historical context found."

    response_content = synthesize_response(prompt, context_snippets)
    save_vector_entry(prompt, timestamp)

    with st.chat_message("assistant"):
        st.markdown(response_content)
    st.session_state.messages.append({"role": "assistant", "content": response_content})
