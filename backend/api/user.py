from fastapi import APIRouter
from core.logger import logger

router = APIRouter()


@router.get("/me")
async def get_current_user_info():
    """获取默认用户信息（不再鉴权）"""
    logger.info(f"👤 获取默认用户信息")
    return {"id": 1, "username": "default_user", "email": "default@example.com"}