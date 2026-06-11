from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base


class RAGFile(Base):
    __tablename__ = "rag_files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # 简化：不再关联users表
    file_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("RAGChunk", back_populates="file")

    def __repr__(self):
        return f"<RAGFile(id={self.id}, file_name={self.file_name}, user_id={self.user_id})>"


class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("rag_files.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chunk_hash = Column(String(64), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    file = relationship("RAGFile", back_populates="chunks")

    def __repr__(self):
        return f"<RAGChunk(id={self.id}, file_id={self.file_id}, chunk_index={self.chunk_index})>"