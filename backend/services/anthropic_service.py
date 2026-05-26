import anthropic
from typing import AsyncGenerator, List, Dict, Any, Optional
from core.config import settings
from core.logger import logger


class AnthropicService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, base_url=settings.ANTHROPIC_API_BASE)
        self.model = settings.ANTHROPIC_MODEL
        logger.info(f"🤖 Anthropic服务初始化 - 模型: {self.model}")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "你是一个专业的RAG智能问答系统，帮助用户解决问答问题。",
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        logger.debug(f"📤 发送到AI的消息数量: {len(messages)}, 工具数量: {len(tools) if tools else 0}")
        try:
            params = {
                "model": self.model,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": messages
            }
            
            if tools and len(tools) > 0:
                params["tools"] = tools
            
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield text
            logger.debug("✅ AI流式响应完成")
        except Exception as e:
            logger.error(f"❌ AI服务错误: {str(e)}")
            yield f"Error: {str(e)}"

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "你是一个专业的RAG智能问答系统，帮助用户解决问答问题。",
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        支持工具调用的同步聊天方法
        返回包含响应内容和工具调用信息的字典
        """
        logger.debug(f"📤 发送带工具的请求到AI, 消息数量: {len(messages)}, 工具数量: {len(tools) if tools else 0}")
        try:
            params = {
                "model": self.model,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": messages
            }
            
            if tools and len(tools) > 0:
                params["tools"] = tools
            
            response = self.client.messages.create(**params)
            
            result = {
                "text": "",
                "tool_calls": []
            }
            
            for content_block in response.content:
                if content_block.type == "text":
                    result["text"] = content_block.text
                elif content_block.type == "tool_use":
                    result["tool_calls"].append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "args": content_block.input
                    })
            
            logger.debug(f"✅ AI响应完成 - 文本: {len(result['text'])}, 工具调用: {len(result['tool_calls'])}")
            return result
        except Exception as e:
            logger.error(f"❌ AI同步服务错误: {str(e)}")
            return {"text": f"Error: {str(e)}", "tool_calls": []}

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "你是一个专业的RAG智能问答系统，帮助用户解决问答问题。"
    ) -> str:
        logger.debug(f"📤 发送同步请求到AI, 消息数量: {len(messages)}")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages
            )
            logger.debug("✅ AI同步响应完成")
            return response.content[0].text
        except Exception as e:
            logger.error(f"❌ AI同步服务错误: {str(e)}")
            return f"Error: {str(e)}"
