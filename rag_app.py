import streamlit as st
from pathlib import Path
import tempfile

from config import SOURCE_DIR, VSTORE_DIR, DOCSTORE_PATH, HASH_FILE
from stores import load_vectorstore, load_docstore, save_docstore, load_hashes, save_hashes
from processing import parse_pdf_elements, get_file_hash, add_documents_to_retriever
from chains import get_text_table_chain, get_image_chain, get_mm_rag_chain

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

# UI
st.set_page_config("Multimodal RAG")
st.title("📄 Multimodal RAG on World Bank Trust Funds Reports")

st.sidebar.header("📁 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

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
st.subheader("🔍 Ask a Question")
question = st.text_input("Enter your question", value="Tell me about major replenishments in FIFs")

if st.button("💬 Get Answer"):
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke(question)
            st.markdown(f"### 🧠 Answer:\n{response}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Save state
vectorstore.persist()
save_docstore(docstore, DOCSTORE_PATH)
save_hashes(file_hashes, HASH_FILE)
