import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests
import threading
from api.main import app as fastapi_app
import uvicorn

def run_fastapi():
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="warning"
    )

# Start FastAPI server in a separate thread
if "fastapi_started" not in st.session_state:
    threading.Thread(target=run_fastapi, daemon=True).start()
    st.session_state["fastapi_started"] = True

# Updated FastAPI endpoint to match the generative model
API_URL = "http://127.0.0.1:8000/generate"

# Page configuration
st.set_page_config(
    page_title="SmolLM Summarizer",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 SmolLM GRPO Summarizer")
st.write("Generate concise summaries using a model fine-tuned with Group Relative Policy Optimization (GRPO).")

# Text input - renamed to 'Prompt' to match the generative task
text_input = st.text_area(
    "Source Document",
    placeholder="Paste a long document here to summarize...",
    height=250
)

# Slider to control generation length, defaulting to the notebook's target
max_tokens = st.slider("Max New Tokens", min_value=10, max_value=96, value=96)

# Generate button
if st.button("Generate Summary"):
    if text_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        try:
            with st.spinner("Generating summary..."):
                # Updated payload: changed "text" to "prompt" and added "max_new_tokens"
                response = requests.post(
                    API_URL,
                    json={
                        "prompt": text_input,
                        "max_new_tokens": max_tokens
                    },
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                summary = result.get("generated_text", "")

                if summary:
                    st.subheader("Generated Summary:")
                    st.success(summary)
                    st.caption(f"Length: {len(summary)} characters")
                else:
                    st.info("The model did not generate a response.")

            else:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Could not connect to the FastAPI server: {e}")