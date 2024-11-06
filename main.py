from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.orm import Session
from database import init_db, SessionLocal, Document
from pydantic import BaseModel
from PyPDF2 import PdfReader, errors as pdf_errors
import os
import json
import requests
from dotenv import load_dotenv
from typing import List
import difflib  # for simple chunk relevance matching
from redis.asyncio import Redis  # Updated Redis import for fastapi-limiter

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Initialize the database
init_db()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize rate limiting middleware
@app.on_event("startup")
async def startup():
    # Initialize Redis client for async Redis support
    redis_client = Redis(host="localhost", port=6379, db=0, decode_responses=True)
    # Initialize FastAPILimiter with Redis client
    await FastAPILimiter.init(redis_client)

# Pydantic models for document response
class DocumentResponse(BaseModel):
    id: int
    filename: str
    upload_date: str

class DocumentListResponse(BaseModel):
    id: int
    filename: str
    upload_date: str

# Model for question-answering request
class QuestionRequest(BaseModel):
    document_id: int
    question: str

# API endpoint to upload a PDF file
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        pdf_reader = PdfReader(file.file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"

        # Create a new document record
        new_document = Document(filename=file.filename, content=text_content)
        db.add(new_document)
        db.commit()
        db.refresh(new_document)
        
        return {
            "id": new_document.id,
            "filename": new_document.filename,
            "upload_date": new_document.upload_date.isoformat()
        }
    except pdf_errors.DependencyError:
        raise HTTPException(status_code=500, detail="PyCryptodome library is required for AES-encrypted PDFs.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# API endpoint to get a document by ID
@app.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": document.id,
        "filename": document.filename,
        "upload_date": document.upload_date.isoformat()
    }

# API endpoint to list all documents
@app.get("/documents", response_model=List[DocumentListResponse])
def list_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).all()
    return [
        {
            "id": document.id,
            "filename": document.filename,
            "upload_date": document.upload_date.isoformat()
        } for document in documents
    ]

# API endpoint to delete a document
@app.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    db.delete(document)
    db.commit()
    return {"message": "Document deleted successfully"}

# HTTP endpoint for question answering with rate limiting
@app.post("/question-answer", dependencies=[Depends(RateLimiter(times=5, seconds=60))])  # Limit to 5 requests per minute
async def question_answer(request: QuestionRequest, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == request.document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    answer = get_answer(request.question, document.content)
    return {"answer": answer}

# WebSocket endpoint for real-time question answering with rate limiting
@app.websocket("/ws/question")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    document_content = None  # Variable to hold document content for context
    db: Session = next(get_db())
    rate_limit_tracker = RateLimiter(times=5, seconds=60)  # Limit to 5 messages per minute
    try:
        while True:
            # Apply rate limit per WebSocket message
            await rate_limit_tracker(websocket)
            data = await websocket.receive_text()
            question_data = json.loads(data)

            if "document_id" in question_data:
                document_id = question_data["document_id"]
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document_content = document.content
                else:
                    await websocket.send_text("Document not found.")
                    continue

            question = question_data.get("question")
            if not document_content:
                await websocket.send_text("No document content available for context.")
                continue

            answer = get_answer(question, document_content)
            await websocket.send_text(answer)
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        db.close()

# Function to get answers using the Hugging Face Inference API
def get_answer(question: str, context: str):
    try:
        context_chunks = split_into_chunks(context)
        relevant_chunk = find_relevant_chunk(question, context_chunks)

        model_name = "distilbert-base-uncased-distilled-squad"
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_TOKEN')}"
        }
        data = {
            "inputs": {
                "question": question,
                "context": relevant_chunk
            }
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
            return answer.replace("\n", " ")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to split context into chunks
def split_into_chunks(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to find the most relevant chunk based on question similarity
def find_relevant_chunk(question, chunks):
    scores = [(chunk, difflib.SequenceMatcher(None, question, chunk).ratio()) for chunk in chunks]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0] if scores else ""
