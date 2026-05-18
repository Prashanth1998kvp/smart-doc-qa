from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

from app.loader import load_and_split
from app.embedder import embed_and_store
from app.retriever import ask_question

app = FastAPI(
    title="Smart Document Q&A API",
    description="Upload a PDF and ask questions about it using RAG + LLM",
    version="1.0.0"
)

UPLOAD_DIR = "uploaded_docs"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Smart Document Q&A API is running",
        "usage": {
            "step_1": "POST /upload — upload a PDF file",
            "step_2": "POST /ask — ask a question about the PDF"
        }
    }


@app.get("/health")
def health():
    chroma_ready = os.path.exists(CHROMA_DIR)
    return {
        "status": "ok",
        "document_indexed": chroma_ready
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    # Validate file is not empty
    if file.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        # Save file to disk
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load, chunk, embed, store
        chunks = load_and_split(file_path)

        if len(chunks) == 0:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from this PDF. It may be scanned or image-based."
            )

        embed_and_store(chunks)

        return {
            "message": "PDF uploaded and indexed successfully",
            "filename": file.filename,
            "total_chunks": len(chunks),
            "status": "ready to answer questions"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


@app.post("/ask")
def ask(request: QuestionRequest):
    # Validate question
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Check if document has been indexed
    if not os.path.exists(CHROMA_DIR):
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet. Please upload a PDF first using POST /upload."
        )

    try:
        result = ask_question(request.question)
        return {
            "question": request.question,
            "answer": result["answer"],
            "source_pages": result["source_pages"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )