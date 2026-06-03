"""
上下文管理器 - 整合GSSC流水线

动态构建模型调用前的最佳上下文，避免上下文腐蚀和注意力分散。
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from services.gsc_pipeline import GSSCPipeline, StructuredContext
from services.memory_storage import memory_storage
from services.qdrant_service import QdrantService
from services.rag_service import RAGService
from core.logger import logger


@dataclass
class ContextBuildResult:
    """上下文构建结果"""
    final_prompt: str
    context_structure: StructuredContext
    token_count: int
    was_compressed: bool
    sources_used: Dict[str, int]  # 各类来源使用的条目数
    sources_tokens: Dict[str, int]  # 各类来源消耗的token数


class ContextManager:
    """
    上下文管理器
    
    整合GSSC流水线，动态构建模型调用前的最佳上下文。
    
    使用流程:
    1. 初始化ContextManager
    2. 调用build_context()构建上下文
    3. 使用返回的prompt调用模型
    """
    
    def __init__(self,
                 user_id: int,
                 max_token_budget: int = 6000,
                 relevance_weight: float = 0.7,
                 recency_weight: float = 0.3,
                 min_relevance_threshold: float = 0.2):
        self.user_id = user_id
        self.max_token_budget = max_token_budget
        
        # 初始化组件
        self.pipeline = GSSCPipeline(
            max_token_budget=max_token_budget,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            min_relevance_threshold=min_relevance_threshold
        )
        
        self.qdrant_service = QdrantService()
        self.rag_service = RAGService(user_id)
        self.memory_storage = memory_storage
        
        # 统计信息
        self.stats = {
            "total_builds": 0,
            "compression_count": 0,
            "avg_tokens_used": 0
        }
    
    def build_context(self,
                    user_query: str,
                    system_prompt: str = "",
                    history: List[Dict[str, str]] = None,
                    enable_rag: bool = True,
                    enable_memory: bool = True,
                    top_k_rag: int = 5,
                    top_k_memory: int = 3,
                    custom_contexts: List[Tuple[str, str]] = None) -> ContextBuildResult:
        """
        构建上下文的入口方法
        
        Args:
            user_query: 用户问题
            system_prompt: 系统提示词（可选，会与GSSC默认策略合并）
            history: 历史对话记录
            enable_rag: 是否启用RAG召回
            enable_memory: 是否启用记忆召回
            top_k_rag: RAG召回数量
            top_k_memory: 记忆召回数量
            custom_contexts: 自定义上下文 [(label, content), ...]
        
        Returns:
            ContextBuildResult: 包含最终prompt和详细信息
        """
        history = history or []
        custom_contexts = custom_contexts or []
        
        logger.info(f"🔄 ContextManager: 开始构建上下文")
        logger.info(f"   用户: {self.user_id}")
        logger.info(f"   查询: {user_query[:50]}...")
        logger.info(f"   RAG: {'启用' if enable_rag else '禁用'}")
        logger.info(f"   记忆: {'启用' if enable_memory else '禁用'}")
        
        # 收集各类上下文
        rag_results = []
        memory_results = []
        
        # RAG召回
        if enable_rag:
            logger.info(f"   📚 正在RAG召回 (top_k={top_k_rag})...")
            try:
                rag_results = self.rag_service.retrieve(user_query, top_k=top_k_rag)
                logger.info(f"   📚 RAG召回完成，获得 {len(rag_results)} 条结果")
            except Exception as e:
                logger.error(f"   ❌ RAG召回失败: {e}")
                rag_results = []
        
        # 记忆召回
        if enable_memory:
            logger.info(f"   🧠 正在记忆召回 (top_k={top_k_memory})...")
            try:
                # 使用新的JSON存储系统快速查询
                memory_results = self._recall_memories(user_query, top_k=top_k_memory)
                logger.info(f"   🧠 记忆召回完成，获得 {len(memory_results)} 条结果")
            except Exception as e:
                logger.error(f"   ❌ 记忆召回失败: {e}")
                memory_results = []
        
        # 执行GSSC流水线
        final_prompt = self.pipeline.run(
            user_query=user_query,
            system_prompt=system_prompt,
            history=history,
            rag_results=rag_results if enable_rag else [],
            memories=memory_results if enable_memory else [],
            custom_contexts=custom_contexts
        )
        
        # 更新统计
        from services.gsc_pipeline import ContextItem
        token_count = ContextItem.estimate_tokens(final_prompt)
        
        self.stats["total_builds"] += 1
        self.stats["avg_tokens_used"] = (
            (self.stats["avg_tokens_used"] * (self.stats["total_builds"] - 1) + token_count)
            / self.stats["total_builds"]
        )
        
        # 构建结果
        result = ContextBuildResult(
            final_prompt=final_prompt,
            context_structure=self.pipeline.structure.structure(
                items=self.pipeline.select.select(
                    self.pipeline.gather.gather(),
                    user_query
                )[0],
                user_query=user_query,
                system_prompt=system_prompt
            ),
            token_count=token_count,
            was_compressed=token_count < self.max_token_budget * 0.7,
            sources_used={
                "system": 1 if system_prompt else 0,
                "history": len(history),
                "rag": len(rag_results),
                "memory": len(memory_results),
                "custom": len(custom_contexts)
            },
            sources_tokens={
                "system": ContextItem.estimate_tokens(system_prompt) if system_prompt else 0,
                "history": sum(ContextItem.estimate_tokens(h.get('content', '')) for h in history),
                "rag": sum(ContextItem.estimate_tokens(r.get('content', '')) for r in rag_results),
                "memory": sum(ContextItem.estimate_tokens(m.get('content', '')) for m in memory_results),
            }
        )
        
        logger.info(f"   ✅ 上下文构建完成")
        logger.info(f"      Token使用: {token_count}")
        logger.info(f"      来源统计: {result.sources_used}")
        
        return result
    
    def _recall_memories(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        召回相关记忆
        
        使用两步查询:
        1. 从JSON索引中快速搜索关键词
        2. 结合向量相似度（如果有Qdrant）
        """
        memories = []
        
        # 1. 从JSON索引获取候选记忆
        query_keywords = self._extract_keywords(query)
        
        candidate_ids = set()
        candidate_scores = {}
        
        # 搜索每个关键词
        for keyword in query_keywords:
            keyword_results = self.memory_storage.search_by_keyword(self.user_id, keyword, limit=top_k * 2)
            for memory in keyword_results:
                memory_id = memory.get('id')
                if memory_id:
                    candidate_ids.add(memory_id)
                    # 基于关键词匹配度评分
                    keyword_match = sum(1 for kw in query_keywords if kw in memory.get('keywords', []))
                    candidate_scores[memory_id] = candidate_scores.get(memory_id, 0) + keyword_match
        
        # 如果没有JSON索引结果，尝试Qdrant
        if not candidate_ids and self.qdrant_service.is_connected():
            try:
                from services.bge_service import bge_service
                query_embedding = bge_service.encode_query(query)
                qdrant_results = self.qdrant_service.search_vectors(
                    query_vector=query_embedding,
                    user_id=self.user_id,
                    limit=top_k,
                    collection_name="memories"
                )
                
                for result in qdrant_results:
                    memory_id = result['payload'].get('memory_id')
                    if memory_id:
                        memory = self.memory_storage.get(self.user_id, memory_id)
                        if memory:
                            memories.append(memory)
            except Exception as e:
                logger.warning(f"⚠️ Qdrant记忆搜索失败: {e}")
        
        # 加载候选记忆并排序
        for memory_id in candidate_ids:
            memory = self.memory_storage.get(self.user_id, memory_id)
            if memory:
                # 结合重要性评分
                importance = memory.get('importance', 1)
                keyword_score = candidate_scores.get(memory_id, 0)
                memory['relevance_score'] = keyword_score + (importance / 10.0)
                memories.append(memory)
        
        # 按相关性评分排序
        memories.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return memories[:top_k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        words = []
        current_word = ""
        
        for char in text:
            if char.isalnum():
                current_word += char
            else:
                if len(current_word) >= 2:
                    words.append(current_word.lower())
                current_word = ""
        
        if len(current_word) >= 2:
            words.append(current_word.lower())
        
        # 去重并返回
        return list(set(words))
    
    def save_memory(self,
                   content: str,
                   memory_type: str = "general",
                   importance: int = 5,
                   conversation_id: Optional[int] = None,
                   metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        保存新记忆
        
        使用结构化JSON存储，自动更新索引
        """
        memory = self.memory_storage.add(
            user_id=self.user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            conversation_id=conversation_id,
            metadata=metadata
        )
        
        # 如果Qdrant可用，也存入向量数据库加速检索
        if self.qdrant_service.is_connected():
            try:
                self._save_memory_to_vector(memory)
            except Exception as e:
                logger.warning(f"⚠️ 记忆存入Qdrant失败: {e}")
        
        return memory
    
    def _save_memory_to_vector(self, memory: Dict[str, Any]) -> None:
        """将记忆存入Qdrant向量数据库"""
        from services.bge_service import bge_service
        
        memory_id = memory.get('id')
        content = memory.get('content', '')
        
        # 生成向量
        embedding = bge_service.encode([content])[0]
        
        # 存入Qdrant
        self.qdrant_service.upsert_memory(
            user_id=self.user_id,
            memory_id=memory_id,
            content=content,
            vector=embedding.tolist(),
            memory_type=memory.get('memory_type'),
            importance=memory.get('importance', 1),
            metadata=memory.get('metadata', {})
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return self.memory_storage.get_stats(self.user_id)
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近记忆"""
        return self.memory_storage.get_recent(self.user_id, limit)
    
    def get_important_memories(self, min_importance: int = 7) -> List[Dict[str, Any]]:
        """获取重要记忆"""
        return self.memory_storage.get_important(self.user_id, min_importance)
    
    def optimize_memory_index(self) -> None:
        """优化记忆索引"""
        self.memory_storage.optimize_index(self.user_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取ContextManager统计信息"""
        return self.stats.copy()


class SimpleContextManager:
    """
    简化版上下文管理器
    
    用于不需要完整GSSC流水线的场景，
    直接构建简单的上下文字符串。
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.context_manager = ContextManager(user_id)
    
    def build(self,
             user_query: str,
             system_prompt: str = "",
             history: List[Dict[str, str]] = None,
             rag_results: List[Dict[str, Any]] = None,
             memories: List[Dict[str, Any]] = None) -> str:
        """
        构建简单上下文（不使用完整GSSC流水线）
        
        适用于快速原型或简单场景
        """
        history = history or []
        rag_results = rag_results or []
        memories = memories or []
        
        parts = []
        
        # 系统提示
        if system_prompt:
            parts.append(f"【系统指令】\n{system_prompt}")
        
        # RAG结果
        if rag_results:
            rag_text = "\n\n".join([
                f"【文档: {r.get('file_name', 'unknown')}】\n{r.get('content', '')}"
                for r in rag_results
            ])
            parts.append(f"【参考文档】\n{rag_text}")
        
        # 记忆
        if memories:
            memory_text = "\n\n".join([
                f"- {m.get('content', '')}"
                for m in memories
            ])
            parts.append(f"【相关记忆】\n{memory_text}")
        
        # 历史对话
        if history:
            history_text = "\n".join([
                f"{h.get('role', 'user')}: {h.get('content', '')}"
                for h in history[-10:]  # 最多10条
            ])
            parts.append(f"【对话历史】\n{history_text}")
        
        # 用户问题
        parts.append(f"【当前问题】\n{user_query}")
        
        return "\n\n".join(parts)
    
    def build_with_gsc(self,
                      user_query: str,
                      system_prompt: str = "",
                      history: List[Dict[str, str]] = None,
                      enable_rag: bool = True,
                      enable_memory: bool = True,
                      top_k_rag: int = 5,
                      top_k_memory: int = 3) -> str:
        """
        使用完整GSSC流水线构建上下文
        """
        result = self.context_manager.build_context(
            user_query=user_query,
            system_prompt=system_prompt,
            history=history,
            enable_rag=enable_rag,
            enable_memory=enable_memory,
            top_k_rag=top_k_rag,
            top_k_memory=top_k_memory
        )
        
        return result.final_prompt
