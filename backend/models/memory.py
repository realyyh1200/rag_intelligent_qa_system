from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base


class UserMemory(Base):
    __tablename__ = "user_memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # 简化：不再关联users表
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True, index=True)
    memory_type = Column(String(50), default="general", index=True)
    content = Column(Text, nullable=False)
    keywords = Column(JSON, nullable=True)
    importance = Column(Integer, default=1)
    access_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime, nullable=True)
    extra_metadata = Column(JSON, default={})
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    conversation = relationship("Conversation", backref="memories")
