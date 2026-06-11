from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from db.database import engine, Base
from api import chat, user, rag
from core.config import settings
from core.logger import logger
import time
import traceback


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 应用启动中...")
    # 初始化 MySQL 数据库（用于会话管理）
    Base.metadata.create_all(bind=engine)
    logger.info("✅ 数据库表初始化完成")
    logger.info("✅ 记忆系统使用 SessionStorage (JSONL)")
    logger.info("✅ RAG 系统使用 ChromaDB")
    yield
    logger.info("👋 应用已关闭")


app = FastAPI(
    title="RAG Smart Question System API",
    description="RAG智能问答系统后端API",
    version="1.0.0",
    lifespan=lifespan
)

# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = f"异常路径: {request.url.path}\n异常类型: {type(exc).__name__}\n异常信息: {str(exc)}\n详细堆栈:\n{traceback.format_exc()}"
    logger.error(f"❌ 全局异常捕获: {error_detail}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"服务器内部错误: {str(exc)}"}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"📥 请求开始: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(f"📤 请求完成: {request.method} {request.url.path} - 状态: {response.status_code} - 耗时: {process_time:.2f}ms")
    return response

app.include_router(chat.router, prefix="/api/chat", tags=["聊天"])
app.include_router(user.router, prefix="/api/users", tags=["用户"])
app.include_router(rag.router, prefix="/api/rag", tags=["RAG"])


@app.get("/")
async def root():
    return {"message": "AI File Processing Website API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
