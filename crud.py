# crud.py
from sqlalchemy.orm import Session
from models import Document

def save_document(db: Session, filename: str, content: str) -> Document:
    """Save document metadata and content to the database."""
    document = Document(filename=filename, content=content)
    db.add(document)
    db.commit()
    db.refresh(document)
    return document

def get_document_by_id(db: Session, document_id: int) -> Document:
    """Retrieve document by ID from the database."""
    return db.query(Document).filter(Document.id == document_id).first()
