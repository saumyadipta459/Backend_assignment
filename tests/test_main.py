import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from main import app

from main import app
from unittest.mock import patch
from io import BytesIO
from fastapi import WebSocketDisconnect
import json


# Initialize TestClient
client = TestClient(app)

# Mocking database session and Redis rate limiter
@pytest.fixture
def mock_db_session():
    # You can mock a database session here
    pass

@pytest.fixture
def mock_redis():
    # You can mock Redis rate limiter here
    pass

# Test for PDF upload functionality
def test_upload_pdf(mock_db_session):
    pdf_file = BytesIO(b"PDF content")
    pdf_file.name = "test.pdf"
    
    response = client.post(
        "/documents/upload", 
        files={"file": ("test.pdf", pdf_file, "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["filename"] == "test.pdf"

# Test for unsupported file type (non-PDF)
def test_upload_unsupported_file(mock_db_session):
    text_file = BytesIO(b"Some text content")
    text_file.name = "test.txt"
    
    response = client.post(
        "/documents/upload", 
        files={"file": ("test.txt", text_file, "text/plain")}
    )
    
    assert response.status_code == 400  # Bad Request
    assert "Error" in response.json().get("detail", "")

# Test for WebSocket connection and message exchange
@pytest.mark.asyncio
async def test_websocket_message():
    async with client.websocket_connect("/ws/question") as websocket:
        # Sending a message
        await websocket.send_text(json.dumps({"document_id": 1, "question": "What is the document about?"}))
        
        # Receiving a response
        response = await websocket.receive_text()
        assert "answer" in response

# Test for question-answering API with rate limiting
def test_question_answer_rate_limiting():
    question_data = {"document_id": 1, "question": "What is the document about?"}
    
    # First request (should pass)
    response = client.post("/question-answer", json=question_data)
    assert response.status_code == 200
    
    # Subsequent requests within rate limit window
    for _ in range(4):  # 5 requests are allowed per minute
        response = client.post("/question-answer", json=question_data)
        assert response.status_code == 200
    
    # 6th request, should hit rate limit
    response = client.post("/question-answer", json=question_data)
    assert response.status_code == 429  # Too Many Requests

# Test for document retrieval by ID
def test_get_document(mock_db_session):
    response = client.get("/documents/1")
    
    assert response.status_code == 200
    assert "filename" in response.json()

# Test for deleting a document
def test_delete_document(mock_db_session):
    response = client.delete("/documents/1")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Document deleted successfully"}

# Test for error handling in missing document (Document not found)
def test_document_not_found(mock_db_session):
    response = client.get("/documents/9999")  # ID 9999 doesn't exist
    
    assert response.status_code == 404
    assert "Document not found" in response.json().get("detail", "")

# Test for WebSocket disconnection
@pytest.mark.asyncio
async def test_websocket_disconnect():
    async with client.websocket_connect("/ws/question") as websocket:
        # Simulate WebSocket disconnect
        await websocket.close()
        with pytest.raises(WebSocketDisconnect):
            await websocket.receive_text()

