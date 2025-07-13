import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

SOURCE_DIR = Path("source_docs")
HASH_FILE = Path("output/hashes.json")
VSTORE_DIR = Path("output/vectorstore")
DOCSTORE_PATH = Path("output/docstore/docstore.pkl")

EMBEDDINGS = OpenAIEmbeddings()
openai_api_key = os.getenv("OPENAI_API_KEY")
