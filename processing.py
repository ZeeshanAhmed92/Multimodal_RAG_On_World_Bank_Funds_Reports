import hashlib
from unstructured.partition.pdf import partition_pdf
from langchain.schema.document import Document
import uuid

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def parse_pdf_elements(filepath):
    chunks = partition_pdf(
        filename=str(filepath),
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    tables, texts, images = [], [], []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Table" in str(type(el)):
                    tables.append(el)
                elif "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)
            texts.append(chunk)
    return texts, tables, images

def add_documents_to_retriever(retriever, elements, summaries, filename, id_key="doc_id"):
    if not elements or not summaries:
        return
    doc_ids = [str(uuid.uuid4()) for _ in elements]
    docs = [Document(page_content=summaries[i], metadata={id_key: doc_ids[i], "source_file": filename}) for i in range(len(elements))]
    retriever.vectorstore.add_documents(docs)
    retriever.docstore.mset(list(zip(doc_ids, elements)))
