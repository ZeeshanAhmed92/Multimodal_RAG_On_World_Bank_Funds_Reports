__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit.components.v1 as components
import streamlit as st
from pathlib import Path
import tempfile
import base64
import time

from config import SOURCE_DIR, VSTORE_DIR, DOCSTORE_PATH, HASH_FILE, CHAT_DATA_DIR
from stores import load_vectorstore, load_docstore, save_docstore, load_hashes, save_hashes, load_chat_history, save_chat_history, generate_session_id
from processing import parse_pdf_elements, get_file_hash, add_documents_to_retriever
from chains import get_text_table_chain, get_image_chain, get_mm_rag_chain

os.environ["LANGCHAIN_TELEMETRY_ENABLED"] = "false"

# Ensure paths exist
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
VSTORE_DIR.mkdir(parents=True, exist_ok=True)
DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
HASH_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load state
vectorstore = load_vectorstore(VSTORE_DIR)
docstore = load_docstore(DOCSTORE_PATH)
retriever = vectorstore.as_retriever()
file_hashes = load_hashes(HASH_FILE)

# Load chains
text_table_chain = get_text_table_chain()
image_chain = get_image_chain()
rag_chain = get_mm_rag_chain(retriever)

# Initialize or restore session
if "chat_id" not in st.session_state:
    st.session_state.chat_id = generate_session_id()

current_chat_path = CHAT_DATA_DIR / f"{st.session_state.chat_id}.json"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(current_chat_path)

# UI
st.set_page_config("Multimodal RAG")

# Title
st.markdown("""
    <h1 style='text-align: center; color: #ffffff; font-weight: bold; margin-top: 10px;'>
        📄 Multimodal RAG on World Bank Trust Funds Reports
    </h1>
""", unsafe_allow_html=True)


# ✅ Background Image Function (Full Page, Clean)
def set_background(image_path):
    if not os.path.exists(image_path):
        st.warning("⚠️ Background image not found.")
        return

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
        <style>
        html, body, .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .stApp {{
            backdrop-filter: blur(0px);
        }}
        </style>
    """, unsafe_allow_html=True)

# ✅ Apply background
set_background("world_bank.jpg")

# 📁 Custom Header in Sidebar (bold + color)
st.sidebar.markdown(
    "<h3 style='color: #000840; font-weight: bold;'>📁 Upload PDF</h3>",
    unsafe_allow_html=True
)

# 📄 Custom Label for File Uploader (bold + color)
st.sidebar.markdown(
    "<span style='font-weight: bold; color: #000840;'>Choose a PDF file</span>",
    unsafe_allow_html=True
)

# Hide default label
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = Path(tmp.name)

    filename = uploaded_file.name
    file_hash = get_file_hash(temp_path)

    if file_hashes.get(filename) == file_hash:
        st.sidebar.success("✅ Already processed.")
    else:
        st.sidebar.info("🔄 Processing...")
        try:
            texts, tables, images = parse_pdf_elements(temp_path)
            text_summaries = text_table_chain.batch(texts, {"max_concurrency": 3})
            table_summaries = text_table_chain.batch([t.metadata.text_as_html for t in tables], {"max_concurrency": 3})
            image_summaries = image_chain.batch(images)

            add_documents_to_retriever(retriever, texts, text_summaries, filename)
            add_documents_to_retriever(retriever, tables, table_summaries, filename)
            add_documents_to_retriever(retriever, images, image_summaries, filename)

            file_hashes[filename] = file_hash
            st.sidebar.success("✅ File processed.")
        except Exception as e:
            st.sidebar.error(f"❌ Failed: {e}")
        finally:
            temp_path.unlink()

# RAG Q&A
# ✅ Styled subheader
st.markdown("""
    <h3 style='font-weight: 700; color: #ffffff;'>
        🔍 Ask a Question
    </h3>
""", unsafe_allow_html=True)

# ✅ Styled label for input
st.markdown("""
    <label style='font-weight: 700; color: #ffffff; font-size: 1rem;'>
        Enter your question
    </label>
""", unsafe_allow_html=True)

# Add this CSS before your input box
st.markdown("""
    <style>
    /* Style the placeholder text */
    input::placeholder {
        color: #000840 !important; /* Light blue */
        opacity: 1; /* Firefox */
    }

    /* Style the actual input text */
    input[type="text"] {
        color: #000840 !important; /* Darker blue for input text */
        font-weight: 500;
        border: 2px solid #000840 !important;
        border-radius: 6px;
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Render the input box (no label)
question = st.text_input("Ask a question", placeholder="Type your question here...", key="question_input", label_visibility="collapsed")

st.markdown("""
    <style>
    .custom-spinner {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 10px;
    }

    .lds-dual-ring {
        display: inline-block;
        width: 24px;
        height: 24px;
    }

    .lds-dual-ring:after {
        content: " ";
        display: block;
        width: 24px;
        height: 24px;
        margin: 1px;
        border-radius: 50%;
        border: 4px solid #28adfe;
        border-color: #28adfe transparent #28adfe transparent;
        animation: lds-dual-ring 1.2s linear infinite;
    }

    @keyframes lds-dual-ring {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }

    .thinking-text {
        font-weight: bold;
        color: #000840;
        font-size: 1.05rem;
    }

    .custom-answer {
        background-color: white;
        color: #000840;
        padding: 1rem;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 1rem;
        border-left: 4px solid #28adfe;
    }

    div.stButton > button {
        background-color: #000840 !important;
        color: white !important;
        font-weight: 700;
        border: 3px solid #28adfe !important;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #28adfe !important;
        color: #000840 !important;
        border-color: #000840 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Button handler
if question and st.button("💬 Get Answer"):
    with st.container():
        spinner_html = """
        <div class="custom-spinner">
            <div class="lds-dual-ring"></div>
            <div class="thinking-text">Thinking...</div>
        </div>
        """
        spinner_slot = st.empty()
        spinner_slot.markdown(spinner_html, unsafe_allow_html=True)

        try:
            response = rag_chain.invoke(question)

            # Add to session + save to disk
            st.session_state.chat_history.append({
                "question": question,
                "answer": response
            })
            save_chat_history(st.session_state.chat_history, current_chat_path)

            spinner_slot.empty()
        except Exception as e:
            spinner_slot.empty()
            st.error(f"❌ Error: {str(e)}")


if st.sidebar.button("🆕 Start New Chat"):
    # Save current chat before resetting
    if st.session_state.get("chat_history"):
        old_path = CHAT_DATA_DIR / f"{st.session_state.chat_id}.json"
        save_chat_history(st.session_state.chat_history, old_path)

    # Reset session
    st.session_state.chat_id = generate_session_id()
    st.session_state.chat_history = []

    st.rerun()

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

user_icon = get_image_base64("boy.png")
bot_icon = get_image_base64("assistant.png")

# Build chat HTML dynamically
chat_html = """
<style>
.scroll-box {
    max-height: 500px;
    overflow-y: auto;
    padding: 12px;
    border-radius: 12px;
    background-color: rgba(255,255,255,0.7);
    backdrop-filter: blur(4px);
    border: 1px solid #ccc;
    margin-top: 1rem;
}

.chat-message {
    display: flex;
    margin-bottom: 12px;
    align-items: flex-start;
}

.chat-message.user {
    justify-content: flex-end;
}

.chat-message.bot {
    justify-content: flex-start;
}

.avatar {
    width: 32px;
    height: 32px;
    margin: 0 8px;
    border-radius: 50%;
}

.bubble {
    padding: 12px 16px;
    border-radius: 18px;
    max-width: 70%;
    font-size: 0.95rem;
    line-height: 1.4;
    word-wrap: break-word;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.bot .bubble {
    background-color: white;
    color: #000;
    border-top-left-radius: 4px;
}

.user .bubble {
    background-color: #d0e8ff;
    color: #000840;
    border-top-right-radius: 4px;
}
</style>

<div class="scroll-box">
"""

# Build chat HTML
for pair in reversed(st.session_state.chat_history):
    # User message (right-aligned)
    chat_html += f"""
    <div class="chat-message user">
        <div class="bubble">{pair['question']}</div>
        <img class="avatar" src="data:image/png;base64,{user_icon}" />
    </div>
    """

    # Bot message (left-aligned)
    chat_html += f"""
    <div class="chat-message bot">
        <img class="avatar" src="data:image/png;base64,{bot_icon}" />
        <div class="bubble">{pair['answer']}</div>
    </div>
    """

# Render chat using components (this respects scrolling)
components.html(chat_html + "</div>", height=520, scrolling=False)


# Save state
save_docstore(docstore, DOCSTORE_PATH)
save_hashes(file_hashes, HASH_FILE)
