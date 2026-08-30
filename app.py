import streamlit as st
import datetime
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_FILE = 'local_vector_index.json'

# Cache the model so it loads once and stays in memory
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

def get_embedding(text):
    vector = model.encode(text)
    return vector.tolist()

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

st.title("🌱 True Semantic RAG Hub")

user_input = st.text_input("Log action or enter query text...")

col1, col2 = st.columns(2)
with col1:
    if st.button("Record & Embed"):
        if user_input.strip():
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            save_vector_entry(user_input.strip(), timestamp)
            st.success("Action recorded with true semantic embedding!")
        else:
            st.error("Entry empty.")

with col2:
    if st.button("Semantic Search"):
        if user_input.strip():
            relevant = query_vector_store(user_input.strip(), top_k=3)
            if relevant:
                st.write(f"--- Semantic Match for: '{user_input}' ---")
                for r in relevant:
                    st.text(f"[{r['time']}] {r['text']}")
            else:
                st.warning("No relevant context found.")
        else:
            st.error("Enter query text.")
