import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
from io import BytesIO
from fastapi import WebSocketDisconnect
import json

# Initialize TestClient
client = TestClient(app)

# Fixture to mock database session
@pytest.fixture
def mock_db_session():
    # Mocking a database session here for use in tests
    pass

# Fixture to mock Redis for rate limiting
@pytest.fixture
def mock_redis():
    # Mocking Redis rate limiter here for use in tests
    pass

# 1. Test for PDF upload functionality
def test_upload_pdf(mock_db_session):
    """Test uploading a valid PDF file."""
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

# 2. Test for unsupported file type (non-PDF)
def test_upload_unsupported_file(mock_db_session):
    """Test uploading an unsupported file type, should return 400 error."""
    text_file = BytesIO(b"Some text content")
    text_file.name = "test.txt"
    
    response = client.post(
        "/documents/upload", 
        files={"file": ("test.txt", text_file, "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Error" in response.json().get("detail", "")

# 3. Test for WebSocket connection and message exchange
@pytest.mark.asyncio
async def test_websocket_message():
    """Test WebSocket connection and message handling."""
    async with client.websocket_connect("/ws/question") as websocket:
        # Send a question message
        await websocket.send_text(json.dumps({"document_id": 1, "question": "What is the document about?"}))
        
        # Receive a response
        response = await websocket.receive_text()
        assert "answer" in response

# 4. Test for question-answering API with rate limiting
def test_question_answer_rate_limiting():
    """Test rate limiting on the question-answer API endpoint."""
    question_data = {"document_id": 1, "question": "What is the document about?"}
    
    # First request (should pass)
    response = client.post("/question-answer", json=question_data)
    assert response.status_code == 200
    
    # Subsequent requests within rate limit window
    for _ in range(4):  # Assuming 5 requests per minute limit
        response = client.post("/question-answer", json=question_data)
        assert response.status_code == 200
    
    # Exceed rate limit
    response = client.post("/question-answer", json=question_data)
    assert response.status_code == 429  # Too Many Requests

# 5. Test for document retrieval by ID
def test_get_document(mock_db_session):
    """Test retrieving a document by its ID."""
    response = client.get("/documents/1")
    
    assert response.status_code == 200
    assert "filename" in response.json()

# 6. Test for deleting a document
def test_delete_document(mock_db_session):
    """Test deleting a document by its ID."""
    response = client.delete("/documents/1")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Document deleted successfully"}

# 7. Test for error handling when document is not found
def test_document_not_found(mock_db_session):
    """Test the response when attempting to retrieve a non-existent document."""
    response = client.get("/documents/9999")  # Assuming ID 9999 does not exist
    
    assert response.status_code == 404
    assert "Document not found" in response.json().get("detail", "")

# 8. Test for WebSocket disconnection handling
@pytest.mark.asyncio
async def test_websocket_disconnect():
    """Test WebSocket disconnection handling."""
    async with client.websocket_connect("/ws/question") as websocket:
        # Close WebSocket connection
        await websocket.close()
        with pytest.raises(WebSocketDisconnect):
            await websocket.receive_text()

# 9. Additional Test for Multiple WebSocket Messages
@pytest.mark.asyncio
async def test_websocket_multiple_messages():
    """Test multiple messages through WebSocket connection."""
    async with client.websocket_connect("/ws/question") as websocket:
        # Send multiple questions
        questions = [
            {"document_id": 1, "question": "What is the content?"},
            {"document_id": 1, "question": "Who wrote this document?"}
        ]
        
        for question in questions:
            await websocket.send_text(json.dumps(question))
            response = await websocket.receive_text()
            assert "answer" in response

# 10. Test for Invalid Document ID during Question Answer
def test_question_answer_invalid_document():
    """Test question-answer API with an invalid document ID."""
    question_data = {"document_id": 9999, "question": "What is this document about?"}  # Assuming ID 9999 does not exist
    
    response = client.post("/question-answer", json=question_data)
    assert response.status_code == 404  # Not Found
    assert "Document not found" in response.json().get("detail", "")

# 11. Test for Invalid Data in Upload Endpoint
def test_invalid_upload_data(mock_db_session):
    """Test upload endpoint with missing file data."""
    response = client.post("/documents/upload", files={})
    assert response.status_code == 422  # Unprocessable Entity

# 12. Test for Empty Question in Question-Answer API
def test_empty_question():
    """Test question-answer API with empty question data."""
    question_data = {"document_id": 1, "question": ""}
    
    response = client.post("/question-answer", json=question_data)
    assert response.status_code == 400  # Bad Request
    assert "Question cannot be empty" in response.json().get("detail", "")
