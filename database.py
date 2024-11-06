# database.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"  # Use appropriate database URL

# Create the database engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# Create the base class for declarative models
Base = declarative_base()

# SessionLocal will be used to create a database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Document model for storing PDF metadata and content
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)  # Store the file name
    content = Column(Text)                 # Store extracted text from the PDF
    upload_date = Column(DateTime, default=datetime.utcnow)  # Store upload timestamp

# Create the database tables
def init_db():
    Base.metadata.create_all(bind=engine)
