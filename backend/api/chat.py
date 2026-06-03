from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import AsyncGenerator
from db.database import get_db
from models.user import User, Conversation, Message
from schemas.schema import ChatRequest, ConversationCreate, ConversationResponse, ConversationUpdate
from core.security import get_current_active_user
from services.anthropic_service import AnthropicService
from services.memory_service import MemoryService
from services.rag_service import RAGService
from services.context_manager import ContextManager
from services.chit_chat_detector import ChitChatDetector
from core.logger import logger
import json

router = APIRouter()
chit_chat_detector = ChitChatDetector()


@router.post("/stream")
async def chat_stream(
    chat_data: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"💬 用户 {current_user.username} 发起聊天请求 (会话ID: {chat_data.conversation_id})")

        if not chat_data.message.strip():
            logger.warning(f"❌ 空消息 - 用户: {current_user.username}")
            raise HTTPException(status_code=400, detail="消息不能为空")

        memory_service = MemoryService(user_id=current_user.id, conversation_id=chat_data.conversation_id)
        
        is_chit_chat = chit_chat_detector.is_chit_chat(chat_data.message)
        
        if not is_chit_chat:
            rag_service = RAGService(user_id=current_user.id)

        conversation = None
        if chat_data.conversation_id:
            conversation = db.query(Conversation).filter(
                Conversation.id == chat_data.conversation_id,
                Conversation.user_id == current_user.id
            ).first()
            if not conversation:
                logger.warning(f"❌ 会话不存在 - ID: {chat_data.conversation_id}")
                raise HTTPException(status_code=404, detail="对话不存在")

        if not conversation:
            conversation = Conversation(
                user_id=current_user.id,
                title=chat_data.message[:50] + "..." if len(chat_data.message) > 50 else chat_data.message
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            logger.info(f"✅ 创建新会话 - ID: {conversation.id}, 用户: {current_user.username}")

        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=chat_data.message
        )
        db.add(user_message)
        db.commit()

        messages_history = []
        messages = db.query(Message).filter(
            Message.conversation_id == conversation.id
        ).order_by(Message.created_at).all()

        for msg in messages:
            messages_history.append({"role": msg.role, "content": msg.content})

        # ========== 使用GSSC上下文管理流水线 ==========
        context_manager = ContextManager(user_id=current_user.id)
        
        # 构建上下文
        context_result = context_manager.build_context(
            user_query=chat_data.message,
            system_prompt=chat_data.system_prompt or "",
            history=messages_history[:-1] if messages_history else [],  # 不含最新一条
            enable_rag=not is_chit_chat,
            enable_memory=True,
            top_k_rag=5,
            top_k_memory=3
        )
        
        logger.info(f"📊 GSSC上下文构建完成 - Token: {context_result.token_count}, 压缩: {context_result.was_compressed}")
        logger.info(f"   来源统计: {context_result.sources_used}")
        
        # 获取构建好的prompt
        enhanced_system_prompt = context_result.final_prompt

        async def stream_response() -> AsyncGenerator[str, None]:
            anthropic_service = AnthropicService()

            full_response = ""
            try:
                logger.info(f"🤖 开始AI响应生成 - 会话ID: {conversation.id}, 闲聊模式: {is_chit_chat}")

                request_messages = [{"role": "user", "content": enhanced_system_prompt}]

                async for chunk in anthropic_service.stream_chat(
                    messages=request_messages,
                    system_prompt=""  # 上下文已包含在user message中
                ):
                    full_response += chunk
                    yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"

                # 保存助手回复
                assistant_message = Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=full_response
                )
                db.add(assistant_message)
                conversation.updated_at = conversation.updated_at
                db.commit()

                # 保存对话到记忆（使用JSON存储）
                memory_service = MemoryService(user_id=current_user.id, conversation_id=conversation.id)
                memory_service.add_message("user", chat_data.message)
                memory_service.add_message("assistant", full_response)
                
                # 同时使用ContextManager保存重要记忆
                if not is_chit_chat and context_result.sources_used.get('rag', 0) > 0:
                    # 如果有RAG结果，保存相关信息到长期记忆
                    context_manager.save_memory(
                        content=f"用户询问: {chat_data.message}\n\nAI回答摘要: {full_response[:500]}...",
                        memory_type="conversation",
                        importance=5,
                        conversation_id=conversation.id
                    )

                logger.info(f"✅ AI响应完成 - 会话ID: {conversation.id}, 响应长度: {len(full_response)}")
                yield f"data: {json.dumps({'content': '', 'done': True, 'conversation_id': conversation.id})}\n\n"

            except Exception as e:
                logger.error(f"❌ AI响应失败 - 会话ID: {conversation.id}, 错误: {str(e)}")
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 聊天流异常: {str(e)}")
        raise HTTPException(status_code=400, detail=f"聊天失败: {str(e)}")


@router.get("/conversations", response_model=list[ConversationResponse])
async def get_conversations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"📋 用户 {current_user.username} 获取会话列表")
        conversations = db.query(Conversation).filter(
            Conversation.user_id == current_user.id
        ).order_by(Conversation.updated_at.desc()).all()
        logger.info(f"✅ 获取会话列表成功 - 数量: {len(conversations)}")
        return conversations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 获取会话列表失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"获取会话列表失败: {str(e)}")


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"📝 用户 {current_user.username} 创建新会话")
        conversation = Conversation(
            user_id=current_user.id,
            title=conversation_data.title or "新对话"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        logger.info(f"✅ 创建会话成功 - ID: {conversation.id}")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 创建会话失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"创建会话失败: {str(e)}")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"🗑️ 用户 {current_user.username} 删除会话 - ID: {conversation_id}")
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            logger.warning(f"❌ 会话不存在 - ID: {conversation_id}")
            raise HTTPException(status_code=404, detail="对话不存在")
        
        db.query(Message).filter(Message.conversation_id == conversation_id).delete()
        db.delete(conversation)
        db.commit()
        logger.info(f"✅ 删除会话成功 - ID: {conversation_id}")
        return {"message": "对话已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 删除会话失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"删除会话失败: {str(e)}")


@router.put("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int,
    conversation_data: ConversationUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"📝 用户 {current_user.username} 更新会话 - ID: {conversation_id}")
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            logger.warning(f"❌ 会话不存在 - ID: {conversation_id}")
            raise HTTPException(status_code=404, detail="对话不存在")
        
        if conversation_data.title is not None:
            conversation.title = conversation_data.title
        
        db.commit()
        db.refresh(conversation)
        logger.info(f"✅ 更新会话成功 - ID: {conversation_id}")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 更新会话失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"更新会话失败: {str(e)}")