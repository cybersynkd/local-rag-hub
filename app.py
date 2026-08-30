import streamlit as st
import datetime
import os
import json
from google import genai

VECTOR_FILE = 'local_vector_index.json'

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

client = genai.Client(api_key=api_key) if api_key else None

def query_vector_store(query_text, top_k=3):
    if not os.path.exists(VECTOR_FILE):
        return []
    try:
        with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
    except Exception:
        return []
        
    query_terms = set(query_text.lower().split())
    scored = []
    for r in records:
        text = r.get('text', '').lower()
        matches = sum(1 for term in query_terms if term in text)
        if matches > 0:
            scored.append((matches, r))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

def save_new_entry(text, timestamp):
    records = []
    if os.path.exists(VECTOR_FILE):
        try:
            with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception:
            records = []
    
    if any(r.get('text') == text for r in records):
        return

    records.append({
        "time": timestamp, 
        "text": text
    })
    with open(VECTOR_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)

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
        context_snippets = "\n".join([f"- [{r.get('time', 'Unknown')}] {r['text']}" for r in relevant])
    else:
        context_snippets = "No prior historical context found."

    response_content = synthesize_response(prompt, context_snippets)
    save_new_entry(prompt, timestamp)

    with st.chat_message("assistant"):
        st.markdown(response_content)
    st.session_state.messages.append({"role": "assistant", "content": response_content})
