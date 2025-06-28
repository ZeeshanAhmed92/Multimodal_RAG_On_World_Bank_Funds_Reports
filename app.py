import os
import uuid
import base64
import streamlit as st
import json
from datetime import datetime
from langchain.embeddings import AzureOpenAIEmbeddings
from run_query_module import (
    run_query, update_faiss_index, extract_text_with_ocr, 
    file_hash, load_existing_hashes, save_hashes, 
    get_session_ids, load_session_messages
)

# Config
PDF_DIR = "./source_docs"
CHAT_HISTORY_DIR = "chat_history"
background_image_path = "Image.jpg"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Azure OpenAI config
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    chunk_size=1000,
)

st.set_page_config(page_title="UTF Story Finder", page_icon="📘", layout="wide")

import os
import base64
import streamlit as st

def set_background(image_path):
    if not os.path.exists(image_path):
        st.warning("⚠️ Background image not found.")
        return

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    html, body, .stApp {{
        height: 100%;
        margin: 0;
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    section[data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.3) !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}
    section[data-testid="stSidebarButton"] * {{
        color: white !important;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 10px;
    }}
    div.stButton > button {{
        background-color: #e6f2ff !important;
        color: #000840 !important;
        font-weight: 700 !important;
        border: 4px solid #000840 !important;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }}
    div.stButton > button:hover {{
        background-color: #d0e8ff !important;
        color: #28adfe !important;
        border-color: #000840 !important;
    }}

    /* ✅ SIDEBAR BUTTONS */
    div[data-testid="stSidebar"] button {{
        background-color: #e6f2ff !important;
        color: #000840 !important;
        font-weight: 700 !important;
        border: 4px solid #000840 !important;
        border-radius: 8px !important;
        padding: 0.6em 1.2em !important;
        transition: 0.3s;
    }}
    div[data-testid="stSidebar"] button:hover {{
        background-color: #d0e8ff !important;
        color: #28adfe !important;
        border-color: #000840 !important;
    }}

    /* 🎯 Specific style for "Start New Chat" (assuming it's the first sidebar button) */
    section[data-testid="stSidebar"] button:nth-of-type(1) {{
        color: #007bff !important; /* Blue text */
        font-weight: 700 !important;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

set_background(background_image_path)

start_chat = st.sidebar.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <form action="">
            <button type="submit" style="
                background-color: #e6f2ff;
                color: #007bff;
                font-weight: bold;
                font-size: 16px;
                padding: 0.6em 1.2em;
                border: 4px solid #000840;
                border-radius: 8px;
                cursor: pointer;
            ">➕ Start New Chat</button>
        </form>
    </div>
""", unsafe_allow_html=True)

if st.query_params.get("") == "":
    st.session_state.clear()
    st.rerun()


if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["chat_messages"] = []

session_id = st.session_state["session_id"]

session_ids = get_session_ids()
selected_session = st.sidebar.selectbox("Select Previous Session", session_ids or ["No sessions yet"])
st.sidebar.markdown("---")

st.markdown("<h3 style='color:#000840;'>📤 Upload UTF Annual Report PDF</h3>", unsafe_allow_html=True)
st.markdown("<label style='color:#000840; font-weight:bold;'>Upload UTF Annual Report</label>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file:
    file_digest = file_hash(uploaded_file)
    existing_hashes = load_existing_hashes()
    st.markdown(f"<div style='background-color:#caebff; color:#000840; padding:10px; border-radius:8px;'>📄 <b>Uploaded:</b> {uploaded_file.name}</div>", unsafe_allow_html=True)

    if file_digest in existing_hashes:
        st.markdown("<div style='background-color:#caebff; color:#000840; padding:10px; border-radius:8px;'>✅ <b>File already processed.</b></div>", unsafe_allow_html=True)
    else:
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner("🔍 Processing with OCR..."):
            extract_text_with_ocr(file_path)
            update_faiss_index(embeddings)

        st.markdown("<div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px;'>✅ <b>Document indexed successfully.</b></div>", unsafe_allow_html=True)

st.markdown("<h3 style='color:#000840 ; font-weight:bold;'>🔍 Ask a Question</h3>", unsafe_allow_html=True)

st.markdown("""
    <style>
    div[data-baseweb="input"] {
        background-color: #e6f2ff !important;
        border: 4px solid #000840 !important; 
        border-radius: 8px !important;
        padding: 5px 10px !important;
        font-weight: bold !important;
    }
    div[data-baseweb="input"] input {
        color: #000840 !important;
        background-color: #e6f2ff !important;
        font-weight: bold !important;
    }
    div[data-baseweb="input"] input::placeholder {
        color: #000840 !important;
        opacity: 0.6 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<label style='color:#000840; font-weight:bold;'>Your question:</label>", unsafe_allow_html=True)
user_question = st.text_input(label="", placeholder="Type your question here...")

st.markdown("""
    <style>
    div.stButton > button {
        background-color: #e6f2ff !important;
        color: #000840 !important;
        font-weight: 700 !important;
        border: 4px solid #000840 !important;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #d0e8ff !important;
        color: #28adfe !important;
        border-color: #000840 !important;
    }
    </style>""", unsafe_allow_html=True)

if st.button("🔎 Get Answer"):
    if not user_question.strip():
        st.markdown("""
            <div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                ⚠️ <b>Please enter a valid question.</b>
            </div>
        """, unsafe_allow_html=True)
    else:
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
            <div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                ⏳ <b>Thinking...</b>
            </div>
        """, unsafe_allow_html=True)

        try:
            answer = run_query(session_id, user_question)
            st.session_state.chat_messages.append({"type": "human", "content": user_question})
            st.session_state.chat_messages.append({"type": "ai", "content": answer})

            with open(os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json"), "w", encoding="utf-8") as f:
                json.dump(st.session_state.chat_messages, f, indent=2)

            thinking_placeholder.empty()
            st.markdown("""
                <div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                    ✅ <b>Answer generated successfully.</b>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            thinking_placeholder.empty()
            st.markdown("""
                <div style='background-color:#ffe6e6; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                    ❌ <b>An error occurred while generating the answer.</b><br>
                    Please try again later.
                </div>
            """, unsafe_allow_html=True)

if st.session_state.get("chat_messages"):
    st.markdown("<h4 style='color:#000840 ; font-weight:bold;'>🧠 AI Answer</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#e6f2ff; color:#000840; padding:12px; border-radius:10px;'>{st.session_state.chat_messages[-1]['content']}</div>", unsafe_allow_html=True)

    st.download_button("📥 Download Answer", st.session_state.chat_messages[-1]['content'], file_name="answer.txt")
    st.download_button("📤 Export Chat JSON", json.dumps(st.session_state.chat_messages, indent=2), file_name=f"{session_id}.json")

if selected_session != "No sessions yet":
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#000840; font-weight:bold;'>🧠 Session Memory Viewer</h4>", unsafe_allow_html=True)

    messages = load_session_messages(selected_session)

    if not messages:
        st.markdown("""
            <div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                ℹ️ <b>No messages found in this session.</b>
            </div>
        """, unsafe_allow_html=True)
    else:
        for msg in messages:
            speaker = "🧑 You" if msg["type"] == "human" else "🤖 AI"
            st.markdown(f"""
                <div style='background-color:#f0f8ff; color:#000840; padding:12px; border-radius:10px; border: 2px solid #000840; margin-bottom:12px;'>
                    <b>{speaker}:</b> {msg['content']}
                </div>
            """, unsafe_allow_html=True)
