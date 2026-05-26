from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6)


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    refresh_token: str


class LoginRequest(BaseModel):
    username: str
    password: str


class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(MessageBase):
    pass


class MessageResponse(MessageBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    title: str = "新对话"


class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(ConversationBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[int] = None
    system_prompt: Optional[str] = "你是一个专业的RAG智能问答系统，帮助用户解决问答问题。"
    file_path: Optional[str] = None  # 用户当前选中的目录路径


# 工具调用相关 Schema
class SafePathCheckRequest(BaseModel):
    """检查路径是否为安全路径的请求"""
    path: str = Field(..., description="需要检查的路径")


class SafePathCheckResponse(BaseModel):
    """检查路径是否为安全路径的响应"""
    is_safe: bool = Field(..., description="路径是否安全")
    message: str = Field(..., description="检查结果说明")


class ToolCallRequest(BaseModel):
    """工具调用请求"""
    tool_name: str = Field(..., description="工具名称")
    tool_args: dict = Field(..., description="工具参数")


# RAG相关 Schema
class RAGProcessRequest(BaseModel):
    """RAG处理请求"""
    paths: List[str] = Field(..., description="文件或文件夹路径列表")


class RAGUploadedFile(BaseModel):
    """上传的文件内容"""
    name: str = Field(..., description="文件名")
    content: str = Field(..., description="文件内容")
    size: int = Field(0, description="文件大小")


class RAGUploadRequest(BaseModel):
    """RAG上传请求"""
    files: List[RAGUploadedFile] = Field(..., description="上传的文件列表")


class RAGProcessResponse(BaseModel):
    """RAG处理响应"""
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="处理结果消息")
    total_files: int = Field(0, description="处理的文件总数")
    total_chunks: int = Field(0, description="生成的chunk总数")
    stored_chunks: int = Field(0, description="成功存储的chunk数")
    skipped_chunks: int = Field(0, description="跳过的chunk数（重复）")
    files: List[str] = Field([], description="处理的文件名列表")


class RAGFileResponse(BaseModel):
    """RAG文件响应"""
    id: int = Field(..., description="文件ID")
    file_name: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(0, description="文件大小")
    chunk_count: int = Field(0, description="Chunk数量")
    created_at: Optional[str] = Field(None, description="创建时间")


class RAGDeleteResponse(BaseModel):
    """RAG删除响应"""
    success: bool = Field(..., description="删除是否成功")
    message: str = Field(..., description="删除结果消息")