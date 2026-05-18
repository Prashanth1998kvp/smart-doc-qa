from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def format_docs(docs):
    # Joins all retrieved chunks into one string for the prompt
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(question: str):
    # Step 1 — Load ChromaDB
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embedding_model()
    )

    # Step 2 — Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Step 3 — Prompt template
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Answer the question using
    only the context provided below. If the answer is not
    in the context, say "I could not find this information
    in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    # Step 4 — LLM
    llm = Ollama(model="llama3")

    # Step 5 — Build the chain using modern pipe syntax
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 6 — Get source pages separately
    source_docs = retriever.invoke(question)
    pages = sorted(set([
        doc.metadata.get("page", "unknown")
        for doc in source_docs
    ]))

    # Step 7 — Run the chain
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "source_pages": pages
    }