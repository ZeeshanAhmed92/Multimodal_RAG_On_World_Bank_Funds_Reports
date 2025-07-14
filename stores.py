import json, pickle
import shutil
from config import EMBEDDINGS
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from chromadb.config import Settings
from chromadb import Client

def load_vectorstore(vstore_dir):
    settings = Settings(
        chroma_db_impl="duckdb",
        persist_directory=str(vstore_dir)
    )
    client = Client(settings)

    try:
        # Return Chroma using the custom client
        return Chroma(
            client=client,
            collection_name="multi_modal_rag",
            embedding_function=EMBEDDINGS
        )
    except Exception:
        # Remove corrupt vectorstore and recreate
        shutil.rmtree(vstore_dir, ignore_errors=True)
        client = Client(settings)  # re-instantiate after deletion
        return Chroma(
            client=client,
            collection_name="multi_modal_rag",
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
