from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(pdf_path: str):
    # Step 1 — Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2 — Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    return chunks