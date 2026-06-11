import openai
from typing import AsyncGenerator, List, Dict, Any, Optional
from core.config import settings
from core.logger import logger


class OpenAIService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.API_KEY, base_url=settings.API_BASE)
        self.model = settings.MODEL
        logger.info(f"🤖 OpenAI服务初始化 - 模型: {self.model}")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "你是一个专业的RAG智能问答系统，帮助用户解决问答问题。",
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        logger.debug(f"📤 发送到AI的消息数量: {len(messages)}, 工具数量: {len(tools) if tools else 0}")
        try:
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)
            
            params = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": formatted_messages,
                "stream": True
            }
            
            if tools and len(tools) > 0:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            
            stream = self.client.chat.completions.create(**params)
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
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
        logger.debug(f"📤 发送带工具的请求到AI, 消息数量: {len(messages)}, 工具数量: {len(tools) if tools else 0}")
        try:
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)
            
            params = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": formatted_messages
            }
            
            if tools and len(tools) > 0:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**params)
            
            result = {
                "text": "",
                "tool_calls": []
            }
            
            if response.choices[0].message.content:
                result["text"] = response.choices[0].message.content
            
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "args": tool_call.function.arguments
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
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)
            
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=formatted_messages
            )
            logger.debug("✅ AI同步响应完成")
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"❌ AI同步服务错误: {str(e)}")
            return f"Error: {str(e)}"