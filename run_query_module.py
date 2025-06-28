# %%
import os
import re
import json
import pickle
import fitz
import uuid
import hashlib
import pytesseract
from PIL import Image
from typing import List
from dotenv import load_dotenv  
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# %%
# Load .env file for Azure keys/config
load_dotenv()

# %%
# Optional: Set path to tesseract executable on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# %%
# Azure OpenAI config
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")  # e.g. text-embedding-3-small
LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")          # e.g. gpt-4-mini

# %%
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") 
# OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# %%
 # Setup Azure Embeddings & LLM
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    chunk_size=1000,  # ✅ 
)

# %%
# embeddings = OpenAIEmbeddings(
#     model=OPENAI_EMBEDDING_MODEL,
#     openai_api_key=OPENAI_API_KEY
# )

# %%
# embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Or "all-minilm" or "bge-base-en"

# %%
# === Path Configs ===
PDF_DIR = "./source_docs"
CHAT_HISTORY_DIR = "chat_history"
FAISS_INDEX_PATH = "./store"  # ✅ Now points directly to where index.faiss is
METADATA_STORE_PATH = "./store/index.pkl"  # ✅ Points to the actual pickle file
HASH_STORE_PATH = "./hashes/index_hashes.txt"
TEXT_CACHE_DIR = "./text_cache"

# %%
def extract_text_with_ocr(pdf_path):
    filename = os.path.basename(pdf_path)
    md_filename = os.path.splitext(filename)[0] + ".md"
    md_path = os.path.join(TEXT_CACHE_DIR, md_filename)

    # If cached .md file exists, read it
    if os.path.exists(md_path):
        print(f"📄 Cached text found for {filename}, loading from Markdown.")
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"🔍 OCR processing: {filename}")
    full_text = ""
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += f"\n\n## Page {page_num + 1} Text\n{text.strip()}"

        try:
            pix = page.get_pixmap(dpi=300)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(image)
            full_text += f"\n\n## Page {page_num + 1} OCR\n{ocr_text.strip()}"
        except Exception as e:
            print(f"⚠️ OCR failed on page {page_num + 1}: {e}")

    # Save as markdown in readable form
    os.makedirs(TEXT_CACHE_DIR, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write(full_text)

    return full_text


# %%
def extract_year(filename):
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else "Unknown"

# %%
def file_hash(file_input):
    """Generate SHA256 hash from UploadedFile, bytes, or file path."""
    h = hashlib.sha256()

    if hasattr(file_input, "getvalue"):  # Streamlit UploadedFile
        h.update(file_input.getvalue())

    elif isinstance(file_input, (bytes, bytearray)):  # raw bytes
        h.update(file_input)

    elif isinstance(file_input, (str, os.PathLike)) and os.path.isfile(file_input):  # path
        with open(file_input, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)

    else:
        raise TypeError("Expected Streamlit UploadedFile, bytes, or valid file path.")

    return h.hexdigest()


def load_existing_hashes():
    """Load file hashes from index_hashes.txt."""
    if not os.path.exists(HASH_STORE_PATH):
        return set()
    with open(HASH_STORE_PATH, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_hashes(hashes: set):
    """Save updated hashes to index_hashes.txt."""
    with open(HASH_STORE_PATH, "w") as f:
        for h in sorted(hashes):
            f.write(f"{h}\n")

def enrich_metadata(filename: str) -> dict:
    year_match = re.search(r"(20\d{2})", filename)
    return {
        "source": filename,
        "year": year_match.group(1) if year_match else "Unknown",
        "fund": "UTF",
        "doc_type": "Annual Report"
    }

def update_faiss_index(embeddings):
    if embeddings is None:
        raise ValueError("❌ Embeddings must be provided to update the FAISS index.")

    print("🔄 Checking for new documents...")

    existing_hashes = load_existing_hashes()
    new_hashes = set()
    new_documents = []

    for filename in os.listdir(PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        file_digest = file_hash(pdf_path)

        if file_digest in existing_hashes:
            print(f"⏭️ Skipping already indexed: {filename}")
            continue

        print(f"📄 New PDF detected: {filename}")
        text = extract_text_with_ocr(pdf_path)
        metadata = enrich_metadata(filename)
        new_documents.append(Document(page_content=text, metadata=metadata))
        new_hashes.add(file_digest)

    if not new_documents:
        print("✅ No new documents found.")
        try:
            return FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"⚠️ Failed to load existing FAISS index: {e}")
            return None

    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    new_chunks = splitter.split_documents(new_documents)

    print(f"📦 Adding {len(new_chunks)} chunks to FAISS vector store...")

    try:
        if os.path.exists(FAISS_INDEX_PATH + ".faiss"):
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(new_chunks)
        else:
            vectorstore = FAISS.from_documents(new_chunks, embeddings)

        vectorstore.save_local(FAISS_INDEX_PATH)
        print("💾 FAISS index saved.")

    except Exception as e:
        print(f"❌ Error during FAISS update: {e}")
        return None

    updated_hashes = existing_hashes.union(new_hashes)
    save_hashes(updated_hashes)
    print(f"✅ Stored {len(updated_hashes)} file hashes in {HASH_STORE_PATH}")

    return vectorstore



# %%
def load_or_create_vectorstore(embeddings):
    return update_faiss_index(embeddings)

# %%
class PersistentChatMessageHistory(ChatMessageHistory):
    def __init__(self, session_id: str):
        super().__init__()
        self._session_id = session_id
        self._file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
        self.load()

    def load(self):
        if os.path.exists(self._file_path):
            with open(self._file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.messages = [self._dict_to_message(msg) for msg in raw]

    def save(self):
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump([self._message_to_dict(msg) for msg in self.messages], f, indent=2)

    def add_message(self, message):
        super().add_message(message)
        self.save()

    def _message_to_dict(self, message):
        return {"type": message.type, "content": message.content}   

    def _dict_to_message(self, data):
        from langchain_core.messages import HumanMessage, AIMessage
        return HumanMessage(content=data["content"]) if data["type"] == "human" else AIMessage(content=data["content"])


# %%
# === Create RAG Chain with Story Extraction Prompt ===
def setup_rag_chain_with_history(session_id: str, embeddings):
    vectorstore = load_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    # llm = ChatOpenAI(
    # model=OPENAI_MODEL,
    # temperature=0,
    # openai_api_key=OPENAI_API_KEY
    # )
    # llm = Ollama(model="llama3.2:latest")  # or any model like "mistral", "phi3", etc.
    llm = AzureChatOpenAI(
        deployment_name=LLM_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.3
    )


    prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI assistant helping users retrieve development results from UTF annual reports.\n\n"
     "Your main goal is to extract and summarize *results stories* when possible.\n\n"
     "Each results story should include:\n"
     "1. A Bold short, descriptive title (5–10 words)\n"
     "2. A summary of the outcome or impact (5–6 sentences) with bold summary title\n"
     "3. Structured metadata:\n"
     "   - **Region**\n"
     "   - **Sector**\n"
     "   - **Donor/Fund**\n"
     "   - **Source Document and Page**\n\n"
     "👉 If you **find stories** related to the user’s question, present them in the structured format above. Make proper headings and make them bold, dont put ## instead of making bold\n"
     "👉 If **no full stories** are available, **fallback to answering the user's question** based on the relevant context from the document dont give random information.\n\n"
     "Be clear and informative. Never retrieve irrelevant information.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: PersistentChatMessageHistory(session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


# %%
# === Run a Query ===
def run_query(session_id: str, question: str):
    rag_chain = setup_rag_chain_with_history(session_id, embeddings)
    result = rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    return result["answer"]

import os

def get_session_ids(chat_history_dir="chat_history"):
    """List all session IDs from saved chat history JSON files."""
    if not os.path.exists(chat_history_dir):
        return []
    return [f.replace(".json", "") for f in os.listdir(chat_history_dir) if f.endswith(".json")]

def load_session_messages(session_id, chat_history_dir="chat_history"):
    """Load all messages from a given session ID JSON file."""
    path = os.path.join(chat_history_dir, f"{session_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


