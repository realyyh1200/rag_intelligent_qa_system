from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from db.database import get_db
from models.user import User
from core.security import get_current_active_user
from services.rag_service import RAGService
from schemas.schema import RAGProcessRequest, RAGProcessResponse, RAGFileResponse, RAGDeleteResponse, RAGUploadRequest
from core.logger import logger
import os

router = APIRouter()


@router.post("/process", response_model=RAGProcessResponse)
async def process_files(
    request: RAGProcessRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """处理文件或文件夹，将内容切分后存入RAG库"""
    try:
        logger.info(f"📥 用户 {current_user.username} 提交RAG处理请求，路径数量: {len(request.paths)}")

        rag_service = RAGService(user_id=current_user.id)
        result = rag_service.process_files(request.paths)

        return result
    except Exception as e:
        logger.error(f"❌ RAG处理失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"处理失败: {str(e)}")


@router.post("/upload", response_model=RAGProcessResponse)
async def upload_files(
    request: RAGUploadRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """处理前端上传的文件内容"""
    try:
        logger.info(f"📥 用户 {current_user.username} 上传RAG文件，数量: {len(request.files)}")

        rag_service = RAGService(user_id=current_user.id)
        result = rag_service.process_uploaded_files([f.dict() for f in request.files])

        return result
    except Exception as e:
        logger.error(f"❌ RAG上传处理失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"处理失败: {str(e)}")


@router.post("/read-file")
async def read_local_file(
    file_path: str,
    current_user: User = Depends(get_current_active_user)
):
    """读取本地文件内容（用于RAG处理）"""
    try:
        logger.info(f"📄 读取本地文件: {file_path}")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")

        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail="路径不是文件")

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        return {
            "success": True,
            "name": file_name,
            "content": content,
            "size": file_size
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 读取文件失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"读取文件失败: {str(e)}")


@router.get("/files", response_model=List[RAGFileResponse])
async def get_rag_files(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """获取用户的RAG文件列表"""
    try:
        rag_service = RAGService(user_id=current_user.id)
        files = rag_service.get_user_files()
        return files
    except Exception as e:
        logger.error(f"❌ 获取RAG文件列表失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"获取失败: {str(e)}")


@router.delete("/files")
async def delete_rag_file(
    file_path: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """删除用户的RAG文件"""
    try:
        rag_service = RAGService(user_id=current_user.id)
        success = rag_service.delete_file(file_path)

        if not success:
            raise HTTPException(status_code=404, detail="文件不存在或无权访问")

        return {"success": True, "message": "删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 删除RAG文件失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"删除失败: {str(e)}")


@router.post("/retrieve")
async def retrieve_documents(
    query: str,
    top_k: int = 5,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """检索相关文档"""
    try:
        rag_service = RAGService(user_id=current_user.id)
        results = rag_service.retrieve(query, top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"❌ RAG检索失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"检索失败: {str(e)}")