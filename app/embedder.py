from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )
    return embedding_model


def embed_and_store(chunks):
    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    return vector_store


def load_existing_store():
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )

    return vector_store