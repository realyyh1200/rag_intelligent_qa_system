"""
记忆服务 - 纯Qdrant存储，移除MySQL依赖
"""
from typing import Dict, List, Optional, Any, Tuple
from services.qdrant_service import QdrantService
from services.bge_service import bge_service
from services.bm25_service import BM25Service
from services.rrf_fusion import RRFusion
from datetime import datetime
from collections import OrderedDict
from core.logger import logger
import json
import uuid


class ShortTermMemory:
    """短期记忆 - 内存存储，用于当前对话"""
    def __init__(self, max_size: int = 10, ttl_seconds: int = 1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._memory: OrderedDict[str, dict] = OrderedDict()

    def add(self, key: str, item: dict) -> None:
        if key in self._memory:
            del self._memory[key]
        item['created_at'] = datetime.utcnow()
        self._memory[key] = item
        if len(self._memory) > self.max_size:
            self._memory.popitem(last=False)

    def get(self, key: str) -> Optional[dict]:
        item = self._memory.get(key)
        if item and self._is_expired(item):
            del self._memory[key]
            return None
        return item

    def get_all(self) -> List[dict]:
        self._cleanup_expired()
        return list(self._memory.values())

    def _is_expired(self, item: dict) -> bool:
        return (datetime.utcnow() - item['created_at']).total_seconds() > self.ttl_seconds

    def _cleanup_expired(self) -> None:
        expired_keys = [k for k, v in self._memory.items() if self._is_expired(v)]
        for k in expired_keys:
            del self._memory[k]

    def clear(self) -> None:
        """清空所有短期记忆"""
        self._memory.clear()


class LongTermMemory:
    """长期记忆 - Qdrant存储"""
    
    def __init__(self, qdrant_service: QdrantService):
        self.qdrant_service = qdrant_service
    
    def add(
        self,
        user_id: int,
        content: str,
        memory_type: str = "general",
        importance: int = 1,
        conversation_id: Optional[int] = None,
        metadata: dict = None
    ) -> Dict[str, Any]:
        """添加长期记忆到Qdrant"""
        memory_id = int(uuid.uuid4().hex[:15], 16)
        embedding = bge_service.encode_query(content)
        
        payload = {
            'user_id': user_id,
            'memory_id': memory_id,
            'memory_type': memory_type,
            'content': content,
            'importance': importance,
            'conversation_id': conversation_id,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        self.qdrant_service.upsert_vector(
            user_id=user_id,
            memory_id=memory_id,
            vector=embedding,
            payload=payload
        )
        
        logger.info(f"LongTermMemory: Added memory id={memory_id}, type={memory_type}")
        return {'id': memory_id, **payload}
    
    def get_user_memories(
        self,
        user_id: int,
        memory_type: str = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取用户的所有记忆"""
        all_memories = self.qdrant_service.get_all_memory_contents_for_user(user_id)
        
        # 过滤memory_type
        if memory_type:
            all_memories = [m for m in all_memories if m.get('memory_type') == memory_type]
        
        # 按重要性排序
        all_memories.sort(key=lambda x: x.get('importance', 1), reverse=True)
        
        return all_memories[:limit]
    
    def get_conversation_memories(self, conversation_id: int, user_id: int) -> List[Dict[str, Any]]:
        """获取特定会话的所有记忆"""
        all_memories = self.qdrant_service.get_all_memory_contents_for_user(user_id)
        return [m for m in all_memories if m.get('conversation_id') == conversation_id]
    
    def delete(self, memory_id: int, user_id: int) -> bool:
        """删除记忆"""
        return self.qdrant_service.delete_vectors(memory_id, user_id)
    
    def clear_conversation_memories(self, conversation_id: int, user_id: int) -> int:
        """清空特定会话的记忆"""
        memories = self.get_conversation_memories(conversation_id, user_id)
        deleted_count = 0
        
        for memory in memories:
            memory_id = memory.get('id')
            if memory_id and self.delete(memory_id, user_id):
                deleted_count += 1
        
        return deleted_count


class HybridRetrievalService:
    """混合检索服务 - 使用RRF融合向量和BM25检索"""
    
    def __init__(self, qdrant_service: QdrantService, user_id: int, alpha: float = 0.7):
        self.qdrant_service = qdrant_service
        self.user_id = user_id
        self.alpha = alpha
        self.bm25 = BM25Service()
        self.rrf_fusion = RRFusion(k=60)  # RRF融合器
    
    def _initialize_bm25(self, memories: List[Dict[str, Any]]) -> None:
        """初始化BM25索引"""
        self.bm25.initialize(memories)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str = None,
        conversation_id: int = None
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """使用RRF融合混合搜索"""
        # 1. 获取所有记忆用于BM25
        all_memories = self.qdrant_service.get_all_memory_contents_for_user(self.user_id)
        
        # 过滤memory_type和conversation_id
        if memory_type:
            all_memories = [m for m in all_memories if m.get('memory_type') == memory_type]
        if conversation_id:
            all_memories = [m for m in all_memories if m.get('conversation_id') == conversation_id]
        
        if not all_memories:
            return []
        
        # 2. BM25检索
        self._initialize_bm25(all_memories)
        bm25_results = self.bm25.search(query, top_k * 3)
        
        # 3. 向量检索
        query_embedding = bge_service.encode_query(query)
        qdrant_results = []
        if self.qdrant_service.is_connected():
            qdrant_results = self.qdrant_service.search_vectors(
                query_vector=query_embedding,
                user_id=self.user_id,
                limit=top_k * 3,
                score_threshold=0.3
            )
        
        # 4. RRF融合
        logger.info(f"🔄 Memory RRF融合开始，BM25: {len(bm25_results)}, 向量: {len(qdrant_results)}")
        
        # 构建用于RRF融合的结果
        bm25_ranked = [(r['id'], i+1) for i, r in enumerate(bm25_results)]
        vector_ranked = []
        for i, r in enumerate(qdrant_results):
            memory_id = r['payload'].get('memory_id')
            if memory_id:
                vector_ranked.append((memory_id, i+1))
        
        # RRF融合
        rankings = [bm25_ranked, vector_ranked]
        fused_scores = self.rrf_fusion.fuse(rankings)
        
        # 5. 构建结果
        memories_dict = {m.get('id'): m for m in all_memories}
        
        results = []
        for memory_id, rrf_score in fused_scores[:top_k]:
            if memory_id in memories_dict:
                memory = memories_dict[memory_id]
                # 确定来源
                source = 'rrf'
                if any(mid == memory_id for mid, _ in bm25_ranked[:top_k]):
                    source = 'bm25'
                elif any(mid == memory_id for mid, _ in vector_ranked[:top_k]):
                    source = 'qdrant'
                
                results.append((memory, rrf_score, source))
        
        return results
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 1,
        conversation_id: Optional[int] = None,
        metadata: dict = None
    ) -> Dict[str, Any]:
        """添加记忆"""
        return LongTermMemory(self.qdrant_service).add(
            user_id=self.user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            conversation_id=conversation_id,
            metadata=metadata
        )


class MemoryService:
    """记忆服务主类 - 完全基于Qdrant"""
    
    SHORT_TERM_MEMORY_MAX_SIZE = 10
    SHORT_TERM_MEMORY_TTL_SECONDS = 1800
    HYBRID_ALPHA = 0.7
    
    def __init__(self, user_id: int, conversation_id: Optional[int] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.qdrant_service = QdrantService()
        
        self.short_term = ShortTermMemory(
            max_size=self.SHORT_TERM_MEMORY_MAX_SIZE,
            ttl_seconds=self.SHORT_TERM_MEMORY_TTL_SECONDS
        )
        self.long_term = LongTermMemory(self.qdrant_service)
        self.hybrid = HybridRetrievalService(self.qdrant_service, user_id, alpha=self.HYBRID_ALPHA)
        
        logger.info(f"MemoryService initialized for user={user_id}, conversation={conversation_id}")
    
    def add_message(self, role: str, content: str) -> None:
        """添加对话消息"""
        key = f"msg_{self.conversation_id}_{datetime.utcnow().timestamp()}"
        item = {
            "content": content,
            "type": "message",
            "role": role,
            "importance": 1
        }
        self.short_term.add(key, item)
        
        # 如果是助手回复，存入长期记忆
        if role == "assistant":
            self.hybrid.add_memory(
                content=content,
                memory_type="learned",
                conversation_id=self.conversation_id,
                importance=1,
                metadata={"role": role}
            )
        
        logger.debug(f"MemoryService: Added message, role={role}")
    
    def add_learned_info(
        self,
        content: str,
        importance: int = 2,
        metadata: dict = None
    ) -> None:
        """添加学习到的信息"""
        self.hybrid.add_memory(
            content=content,
            memory_type="learned",
            conversation_id=self.conversation_id,
            importance=importance,
            metadata=metadata
        )
        logger.info(f"MemoryService: Stored learned info, importance={importance}")
    
    def get_context_for_ai(self, include_recent: int = 5, hybrid_top_k: int = 3) -> str:
        """为AI生成上下文"""
        context_parts = []
        
        # 短期记忆
        context_parts.append("## 短期记忆 (当前对话)")
        recent_items = self.short_term.get_all()[-include_recent:]
        if recent_items:
            for i, item in enumerate(recent_items, 1):
                role = item.get('role', 'unknown')
                content = item.get('content', '')
                context_parts.append(f"{i}. [{role}] {content}")
        else:
            context_parts.append("(无)")
        
        # 长期记忆
        context_parts.append("\n## 长期记忆 (语义检索)")
        query = " ".join([item['content'] for item in recent_items]) if recent_items else "对话 助手 AI"
        hybrid_results = self.hybrid.search(
            query=query,
            top_k=hybrid_top_k,
            memory_type="learned",
            conversation_id=self.conversation_id
        )
        
        if hybrid_results:
            for memory, score, source in hybrid_results:
                context_parts.append(f"- [{source}] {memory.get('content', '')}")
        else:
            long_term_memories = self.long_term.get_user_memories(
                self.user_id, memory_type="learned", limit=5
            )
            if long_term_memories:
                for mem in long_term_memories:
                    context_parts.append(f"- [importance:{mem.get('importance', 1)}] {mem.get('content', '')}")
            else:
                context_parts.append("(无)")
        
        result = "\n".join(context_parts)
        logger.debug(f"MemoryService: Generated context, length={len(result)}")
        return result
    
    def store_user_preference(self, key: str, value: Any) -> None:
        """存储用户偏好"""
        content = json.dumps({"key": key, "value": value})
        self.hybrid.add_memory(
            content=content,
            memory_type="preference",
            conversation_id=None,
            importance=3,
            metadata={"preference_key": key}
        )
        logger.info(f"MemoryService: Stored preference {key}={value}")
    
    def get_user_preferences(self) -> List[Dict[str, Any]]:
        """获取用户偏好"""
        return self.long_term.get_user_memories(self.user_id, memory_type="preference")
    
    def clear_conversation_memory(self) -> None:
        """清空当前会话的记忆"""
        if self.conversation_id:
            self.short_term.clear()
            deleted_count = self.long_term.clear_conversation_memories(self.conversation_id, self.user_id)
            logger.info(f"MemoryService: Cleared memory for conversation={self.conversation_id}, deleted {deleted_count} memories")
    
    def get_memory_stats(self) -> dict:
        """获取记忆统计信息"""
        memory_count = self.qdrant_service.count_vectors(self.user_id)
        return {
            "short_term_count": len(self.short_term.get_all()),
            "long_term_memories": memory_count,
            "bge_embedding_dim": bge_service.embedding_dim
        }
    
    def delete_all_memories(self) -> bool:
        """删除用户的所有记忆"""
        return self.qdrant_service.delete_all_user_memories(self.user_id)
