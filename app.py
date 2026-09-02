import streamlit as st
import datetime
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai

VECTOR_FILE = 'local_vector_index.json'

st.set_page_config(
    page_title="Personal AI & Benchmark Hub",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Key Setup
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

client = genai.Client(api_key=api_key) if api_key else None

# Vector Store Functions
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

# Sidebar Navigation
app_mode = st.sidebar.selectbox(
    "Navigation Hub",
    ["💬 Personal Intelligence Chat", "📊 AI Model Comparison Dashboard"]
)

if app_mode == "💬 Personal Intelligence Chat":
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

elif app_mode == "📊 AI Model Comparison Dashboard":
    st.title("🤖 AI Response Comparison Dashboard")
    st.markdown("""
    Comparing how four different AIs answered:
    1. **"Why are you the best choice?"**
    2. **"Best way to test models?"**
    3. **"What's holding you back?"**
    """)

    models = ["Personal AI (Streamlit-style)", "Gemini", "Grok", "Claude"]

    strengths_data = {
        "Model": models,
        "Truth-seeking / Honesty": [7.0, 6.0, 9.5, 8.0],
        "Personalization / Memory": [9.5, 8.0, 6.0, 5.5],
        "Less Censored / Openness": [6.0, 5.5, 9.5, 7.0],
        "Reasoning / Depth": [7.5, 8.0, 8.5, 9.0],
        "Personality / Humor": [6.0, 5.0, 9.0, 6.5],
        "Speed / Efficiency Claims": [7.0, 9.0, 7.0, 6.5],
        "Versatility Claims": [8.5, 7.5, 8.0, 8.5],
        "Self-awareness of Limits": [7.0, 6.5, 9.0, 9.0],
    }

    df_str = pd.DataFrame(strengths_data)

    limitations_data = {
        "Model": models,
        "Physical Agency": [8, 7, 8, 8],
        "Knowledge Cutoff": [6, 6, 8, 9],
        "Hallucination Risk": [5, 5, 8, 9],
        "No Consciousness": [9, 7, 8, 8],
        "Safety Rails Impact": [7, 6, 8, 7],
        "Context Limits": [4, 6, 7, 8],
    }

    df_lim = pd.DataFrame(limitations_data)

    st.sidebar.header("Dashboard Controls")
    view = st.sidebar.radio(
        "Select View",
        ["Overview", "Radar Chart", "Bar Comparisons", "Limitations Heatmap", "Text Summary", "Raw Scores"]
    )

    color_map = {
        "Personal AI (Streamlit-style)": "#FF6B6B",
        "Gemini": "#4ECDC4",
        "Grok": "#45B7D1",
        "Claude": "#96CEB4"
    }

    if view == "Overview":
        st.header("High-Level Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Personal AI", "Memory Focus", "Strong personalization claims")
            st.info("Emphasizes continuity, partnership, and structured self-marketing.")
        
        with col2:
            st.metric("Gemini", "Speed Focus", "Mobile-optimized Flash")
            st.info("Highlights efficiency, paid-tier power, and technical constraints.")
        
        with col3:
            st.metric("Grok", "Truth Focus", "Less censored")
            st.success("Direct truth-seeking, humor, minimal PR safety theater.")
        
        with col4:
            st.metric("Claude", "Caution Focus", "Humble & precise")
            st.info("Measured, admits limits clearly, prioritizes real-task testing.")
        
        st.markdown("---")
        st.subheader("Key Takeaways from the Texts")
        
        st.markdown("""
        | Aspect | Personal AI | Gemini | Grok | Claude |
        |--------|-------------|--------|------|--------|
        | **Main pitch** | Memory + Partnership | Speed + Mobile | Truth-seeking | Honesty + Caution |
        | **Strongest claim** | Context continuity | Flash speed | Less censored | Nuanced reasoning |
        | **Testing advice** | Very detailed methods | Arena + benchmarks | Custom + Arena | Real tasks first |
        | **Limits style** | Numbered structured | Technical | Direct comprehensive | Humble specific |
        | **Tone** | Marketing-ish | Product-focused | Direct / witty | Measured / humble |
        """)

    elif view == "Radar Chart":
        st.header("Strength Claims – Radar Chart")
        
        categories = [c for c in df_str.columns if c != "Model"]
        
        fig = go.Figure()
        
        for model in models:
            row = df_str[df_str["Model"] == model].iloc[0]
            values = [row[c] for c in categories]
            values += values[:1]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=model,
                line_color=color_map[model],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            title="Strength Claims Comparison (1-10 scale based on response content)",
            height=650
        )
        
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Bar Comparisons":
        st.header("Key Differentiator Bar Charts")
        
        key_dims = [
            "Truth-seeking / Honesty",
            "Personalization / Memory",
            "Less Censored / Openness",
            "Self-awareness of Limits"
        ]
        
        for dim in key_dims:
            fig = px.bar(
                df_str,
                x="Model",
                y=dim,
                color="Model",
                color_discrete_map=color_map,
                title=dim,
                text=dim
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(yaxis_range=[0, 10.5], showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif view == "Limitations Heatmap":
        st.header("Transparency on Limitations")
        st.markdown("Higher score = the model more openly and thoroughly discussed this limitation in its response.")
        
        df_heat = df_lim.set_index("Model")
        
        fig = px.imshow(
            df_heat,
            text_auto=True,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="How openly each model discussed its limitations"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Text Summary":
        st.header("Qualitative Summary")
        
        with st.expander("Personal AI (Streamlit-style)", expanded=True):
            st.markdown("""
            **Strengths emphasized**: Context & memory continuity, versatility, accuracy, dedicated partnership.
            **Testing advice**: Very detailed – Blind A/B, Golden Dataset, LLM-as-Judge, public benchmarks, operational factors.
            **Limitations**: Physical agency, dependence on context, no true human experience, safety constraints.
            """)
        
        with st.expander("Gemini"):
            st.markdown("""
            **Strengths emphasized**: Speed (Flash-Lite / mobile), paid-tier power, deep personalization.
            **Testing advice**: Chatbot Arena, standardized benchmarks, golden prompts, LLM-as-Judge.
            **Limitations**: Context window cost, real-time tool latency, nuance in open-ended reasoning.
            """)
        
        with st.expander("Grok"):
            st.markdown("""
            **Strengths emphasized**: Maximum truth-seeking, less censored, curious & helpful, humor.
            **Testing advice**: Clear criteria → public leaderboards + personal side-by-side testing.
            **Limitations**: Hallucinations, knowledge cutoff, reasoning depth limits, compute trade-offs.
            """)
        
        with st.expander("Claude"):
            st.markdown("""
            **Strengths emphasized**: Reasoning, writing, coding, analysis; tries to be direct and honest.
            **Testing advice**: Preference for testing on *your actual tasks*; LMArena + benchmarks as secondary.
            **Limitations**: Knowledge cutoff, no persistent memory by default, can be confidently wrong.
            """)

    else:
        st.header("Raw Scores")
        st.subheader("Strength Scores (1-10)")
        st.dataframe(df_str.set_index("Model").style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
        
        st.subheader("Limitation Transparency Scores (1-10)")
        st.dataframe(df_lim.set_index("Model").style.background_gradient(cmap="Oranges", axis=None), use_container_width=True)
