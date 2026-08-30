import streamlit as st
import datetime
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

VECTOR_FILE = 'local_vector_index.json'

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)

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
    records.append({
        "time": timestamp, 
        "text": text, 
        "vector": get_embedding(text)
    })
    with open(VECTOR_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)

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
    try:
        model_llm = genai.GenerativeModel('gemini-3.6-flash')
        full_prompt = f"""You are a personal intelligence assistant. Use the following retrieved historical context to answer the user's prompt accurately. If the context doesn't have the answer, rely on your general knowledge while keeping the user's records in mind.

Retrieved Context:
{context_snippets}

User Prompt: {prompt}
"""
        response = model_llm.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Logged to memory, but synthesis failed: {str(e)}"

st.title("🌱 Personal Intelligence Chat Hub")

# Sidebar for File Ingestion
st.sidebar.header("Bulk Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload text or code file", type=["txt", "py", "md", "json"])
if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    # Chunk file content by paragraphs or lines if large, or save as single entry
    save_vector_entry(f"File Upload [{uploaded_file.name}]:\n{file_content}", timestamp)
    st.sidebar.success(f"Successfully ingested {uploaded_file.name} into vector store!")

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
