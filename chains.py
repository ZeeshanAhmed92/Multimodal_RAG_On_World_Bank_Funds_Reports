from base64 import b64decode
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI as VisionModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage

def get_text_table_chain():
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    Respond only with the summary, no additional comment.
    Table or text chunk: {element}
    """)
    model = ChatOpenAI(temperature=0.5, model="gpt-4.1-mini")
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

def get_image_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": "Describe the image in detail. It’s from a trust fund report."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
        ])
    ])
    model = VisionModel(model="gpt-4.1-mini")
    return prompt | model | StrOutputParser()

import imghdr
from base64 import b64decode
from langchain.docstore.document import Document

def is_supported_image(b64_string):
    try:
        image_data = b64decode(b64_string)
        image_type = imghdr.what(None, h=image_data)
        return image_type in ["jpeg", "png", "gif", "webp"], image_type
    except Exception:
        return False, None

def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        content = doc.page_content
        is_image, image_type = is_supported_image(content)
        if is_image:
            b64.append((content, image_type))  # Store tuple (b64, mime)
        else:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = "".join([t.page_content for t in docs_by_type["texts"]])

    prompt_instructions = f"""
You are an expert AI assistant.

Answer the user's question using only the information provided in the context below, which may include text, tables, and image summaries.

- If the context contains enough information to answer the question clearly, provide a direct, accurate, and slightly elaborated response.
- If the context lacks sufficient information or the question is too vague or ambiguous, **do not make assumptions**. Instead, ask a **clear, specific follow-up question** to help the user clarify their intent.

Respond accordingly.

Context:
{context_text}

Question:
{user_question}
""".strip()

    prompt_content = [{"type": "text", "text": prompt_instructions}]
    for image_b64, image_type in docs_by_type["images"]:
        mime = f"image/{'jpeg' if image_type == 'jpg' else image_type}"
        prompt_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:{mime};base64,{image_b64}"}
        })

    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])



def get_mm_rag_chain(retriever):
    return (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
