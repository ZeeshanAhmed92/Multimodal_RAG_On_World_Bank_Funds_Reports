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

def parse_docs(docs):
    from base64 import b64decode
    b64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = "".join([t.page_content for t in docs_by_type["texts"]])

    prompt_content = [{"type": "text", "text": f"""
    Answer the question based only on the following context, which may include text, tables, and images.
    Context: {context_text}
    Question: {user_question}
    """.strip()}]
    for image in docs_by_type["images"]:
        prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
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
