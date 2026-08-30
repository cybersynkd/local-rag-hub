import streamlit as st
import datetime
import os
import json
import hashlib

VECTOR_FILE = 'local_vector_index.json'

def simple_embed(text):
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def save_vector_entry(text, timestamp):
    records = []
    if os.path.exists(VECTOR_FILE):
        try:
            with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception:
            records = []
    records.append({"time": timestamp, "text": text, "vector": simple_embed(text)})
    with open(VECTOR_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)

def query_vector_store(query_text, top_k=3):
    if not os.path.exists(VECTOR_FILE):
        return []
    with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)
    q_vec = simple_embed(query_text)
    scored = [(cosine_similarity(q_vec, r.get('vector', simple_embed(r.get('text', '')))), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

st.title("🌱 Local Vector RAG Hub")

user_input = st.text_input("Log action or enter query text...")

col1, col2 = st.columns(2)
with col1:
    if st.button("Record & Embed"):
        if user_input.strip():
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            save_vector_entry(user_input.strip(), timestamp)
            st.success("Action recorded & embedded!")
        else:
            st.error("Entry empty.")

with col2:
    if st.button("Vector Search"):
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
