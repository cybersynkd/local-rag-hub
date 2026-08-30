import streamlit as st
import datetime
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_FILE = 'local_vector_index.json'

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

def get_embedding(text):
    return model.encode(text).tolist()

def cosine_similarity(v1, v2):
    a, b = np.array(v1), np.array(v2)
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
        vec = r.get('vector', get_embedding(r.get('text', '')))
        score = cosine_similarity(q_vec, vec)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

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
        response_content = f"Logged to memory. Relevant context found:\n{context_snippets}"
    else:
        response_content = f"Logged to memory. No prior matching context found for '{prompt}'."

    save_vector_entry(prompt, timestamp)

    with st.chat_message("assistant"):
        st.markdown(response_content)
    st.session_state.messages.append({"role": "assistant", "content": response_content})
