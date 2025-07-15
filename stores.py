import json, pickle
import shutil
import os
from datetime import datetime
import uuid
from config import EMBEDDINGS
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore

def load_vectorstore(vstore_dir):
    if not any(vstore_dir.glob("*")):
        return Chroma(
            collection_name="multi_modal_rag",
            persist_directory=str(vstore_dir),
            embedding_function=EMBEDDINGS
        )
    try:
        return Chroma(
            collection_name="multi_modal_rag",
            persist_directory=str(vstore_dir),
            embedding_function=EMBEDDINGS
        )
    except Exception:
        shutil.rmtree(vstore_dir, ignore_errors=True)
        return Chroma(
            collection_name="multi_modal_rag",
            persist_directory=str(vstore_dir),
            embedding_function=EMBEDDINGS
        )


def load_docstore(docstore_path):
    if docstore_path.exists():
        with open(docstore_path, "rb") as f:
            return pickle.load(f)
    return InMemoryStore()

def save_docstore(docstore, docstore_path):
    with open(docstore_path, "wb") as f:
        pickle.dump(docstore, f)

def load_hashes(json_path):
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hashes, json_path):
    with open(json_path, "w") as f:
        json.dump(hashes, f, indent=2)

def save_chat_history(chat, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat, f, indent=2)

def load_chat_history(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(chat, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat, f, indent=2)

def load_chat_history(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def generate_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
