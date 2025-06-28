import os
import uuid
import base64
import streamlit as st
import json
from run_query_module import (
    run_query, update_faiss_index, extract_text_with_ocr, 
    file_hash, load_existing_hashes, save_hashes, 
    get_session_ids, load_session_messages
)

# Config
PDF_DIR = "./source_docs"
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

st.set_page_config(page_title="UTF Story Finder", page_icon="📘", layout="wide")
# -------------------- Set Background Image --------------------
def set_background(image_path):
    """Set a full-page background image and apply 60% transparency to the sidebar background."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    /* Background image for full app */
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Sidebar with 60% transparent dark overlay */
    section[data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.3) !important;
    }}

    /* Optional: make sidebar content white for contrast */
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    /* Optional: main block slight overlay for readability */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
# -------------------- Run App --------------------
# Path to your image file (make sure it's in the same folder or give full path)
background_image_path = "Image.jpg"
set_background(background_image_path)

st.markdown("<h1 style='text-align: center; color: #000840  ;'>📘 UTF Results Story Finder</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Options")

session_ids = get_session_ids()
selected_session = st.sidebar.selectbox("Select Session", session_ids or ["No sessions yet"])
st.sidebar.markdown("---")

# Upload
st.markdown(
    "<h3 style='color:#000840 ;'>📤 Upload UTF Annual Report PDF</h3>", 
    unsafe_allow_html=True
)

# Custom styled uploader label using markdown
st.markdown(
    "<label style='color:#000840; font-weight:bold;'>Upload UTF Annual Report</label>",
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file:
    file_digest = file_hash(uploaded_file)  # Compute hash directly from file bytes
    existing_hashes = load_existing_hashes()

    # Show uploaded filename once
    st.markdown(
        f"<div style='background-color:#caebff; color:#000840 ; padding:10px; border-radius:8px;'>"
        f"📄 <b>Uploaded:</b> {uploaded_file.name}"
        "</div>",
        unsafe_allow_html=True
    )

    if file_digest in existing_hashes:
        st.markdown(
            "<div style='background-color:#caebff; color:#000840 ; padding:10px; border-radius:8px;'>"
            "✅ <b>File already processed.</b>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        # Save the file after confirming it's new
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner("🔍 Processing with OCR..."):
            extract_text_with_ocr(file_path)
            update_faiss_index(embeddings=None)

        st.markdown(
            "<div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px;'>"
            "✅ <b>Document indexed successfully.</b>"
            "</div>",
            unsafe_allow_html=True
        )

# Input Section Header
st.markdown(
        "<h3 style='color:#000840 ; font-weight:bold;'>🔍 Ask a Question</h3>",
    unsafe_allow_html=True
)

# Input Field (no default text)
# Inject custom CSS for light blue text input
st.markdown("""
    <style>
    /* Style the outer container of the input */
    div[data-baseweb="input"] {
        background-color: #e6f2ff !important;
        border: 4px solid #000840 !important; 
        border-radius: 8px !important;
        padding: 5px 10px !important;
        font-weight: bold !important;
    }

    /* Style the actual <input> element inside */
    div[data-baseweb="input"] input {
        color: #000840 !important;
        background-color: #e6f2ff !important;
        font-weight: bold !important;
    }

    /* Placeholder styling */
    div[data-baseweb="input"] input::placeholder {
        color: #000840 !important;
        opacity: 0.6 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<label style='color:#000840; font-weight:bold;'>Your question:</label>", unsafe_allow_html=True)
user_question = st.text_input(label="", placeholder="Type your question here...")

# Ask Button
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #e6f2ff !important;
        color: #000840 !important;
        font-weight: 700 !important;          /* Stronger bold */
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
        # 🔶 Custom Warning (Blue-themed)
        st.markdown("""
            <div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                ⚠️ <b>Please enter a valid question.</b>
            </div>
        """, unsafe_allow_html=True)

    else:
        new_session_id = f"session_{uuid.uuid4().hex[:8]}"

        # ⏳ Custom Thinking Spinner (Blue-themed)
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
            <div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                ⏳ <b>Thinking...</b>
            </div>
        """, unsafe_allow_html=True)

        try:
            answer = run_query(new_session_id, user_question)
            st.session_state["last_answer"] = answer
            st.session_state["last_session"] = new_session_id

            thinking_placeholder.empty()  # Clear spinner

            # ✅ Custom Success
            st.markdown("""
                <div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                    ✅ <b>Answer generated successfully.</b>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            thinking_placeholder.empty()

            # ❌ Custom Error (Red-themed)
            st.markdown("""
                <div style='background-color:#ffe6e6; color:#000840 ; padding:10px; border-radius:8px; border:2px solid #000840 ;'>
                    ❌ <b>An error occurred while generating the answer.</b><br>
                    Please try again later.
                </div>
            """, unsafe_allow_html=True)

# Output Area
if "last_answer" in st.session_state:
    st.markdown(
        "<h4 style='color:#000840 ; font-weight:bold;'>🧠 AI Answer</h4>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='background-color:#e6f2ff; color:#000840; padding:12px; border-radius:10px;'>"
        f"{st.session_state['last_answer']}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <style>
    /* Custom download button */
    .stDownloadButton button {
        background-color: #e6f2ff !important;
        color: #000840  !important;
        font-weight: bold !important;
        border: 3px solid #000840  !important;
        border-radius: 8px !important;
        padding: 0.6em 1.2em !important;
        transition: 0.3s ease;
    }
    .stDownloadButton button:hover {
        background-color: #d0e8ff !important;
        color: #28adfe !important;
        border-color: #000840  !important;
    }
    </style>
""", unsafe_allow_html=True)

    st.download_button("📥 Download Answer", st.session_state["last_answer"], file_name="answer.txt")

# Memory Viewer Section
if selected_session != "No sessions yet":
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='color:#000840; font-weight:bold;'>🧠 Session Memory Viewer</h4>",
        unsafe_allow_html=True
    )

    messages = load_session_messages(selected_session)

    if not messages:
        st.markdown("""
            <div style='background-color:#e6f2ff; color:#000840; padding:10px; border-radius:8px; border:2px solid #000840;'>
                ℹ️ <b>No messages found in this session.</b>
            </div>
        """, unsafe_allow_html=True)
    else:
        for i, msg in enumerate(messages):
            speaker = "🧑 You" if msg["type"] == "human" else "🤖 AI"
            st.markdown(
                f"""
                <div style='background-color:#f0f8ff; color:#000840; 
                            padding:12px; border-radius:10px; 
                            border: 2px solid #000840 ; 
                            margin-bottom:12px;'>
                    <b>{speaker}:</b> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
