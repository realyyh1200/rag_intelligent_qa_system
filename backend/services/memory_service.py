"""
记忆服务 - 基于 SessionStorage (JSONL) 架构
采用 OpenClaw 的 "Index(JSON) + Log(JSONL)" 双层存储
使用 sentence-transformers 实现词嵌入和精排功能
"""
from typing import Dict, List, Optional, Any, Tuple
from services.chroma_service import ChromaService
from services.embedding_service import embedding_service
from services.bm25_service import BM25Service
from services.rrf_fusion import RRFusion
from services.session_storage import SessionStorage
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
    """长期记忆 - SessionStorage (JSONL) 存储"""

    def __init__(self, session_storage: SessionStorage):
        self.session_storage = session_storage

    def add(
        self,
        user_id: int,
        content: str,
        memory_type: str = "general",
        importance: int = 1,
        conversation_id: Optional[int] = None,
        metadata: dict = None
    ) -> Dict[str, Any]:
        """添加长期记忆到 SessionStorage"""
        # 获取或创建会话
        session = self.session_storage.get_conversation_session(user_id, conversation_id)
        if not session:
            session = self.session_storage.create_session(
                user_id=user_id,
                conversation_id=conversation_id,
                model="default",
                tags=[]
            )

        # 添加记忆到会话
        self.session_storage.append_memory(
            user_id=user_id,
            session_id=session["id"],
            content=content,
            memory_type=memory_type,
            importance=importance
        )

        # 同时存储到 ChromaDB 用于向量检索
        chroma_service = ChromaService()
        memory_id = int(uuid.uuid4().hex[:15], 16)

        chroma_service.upsert_memory(
            user_id=user_id,
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            conversation_id=conversation_id,
            metadata={'created_at': datetime.utcnow().isoformat(), **(metadata or {})}
        )

        logger.info(f"LongTermMemory: Added memory to session {session['id']}, id={memory_id}, type={memory_type}")
        return {'id': memory_id, 'session_id': session['id'], 'content': content, 'memory_type': memory_type, 'importance': importance, 'conversation_id': conversation_id}

    def get_user_memories(
        self,
        user_id: int,
        memory_type: str = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取用户的所有记忆"""
        # 从所有会话中获取记忆
        all_memories = []
        index = self.session_storage._get_index_path(user_id)

        if not index.exists():
            return []

        try:
            with open(index, 'r', encoding='utf-8') as f:
                sessions_data = json.load(f)

            for session_info in sessions_data.get("sessions", []):
                session_id = session_info["id"]
                memories = self.session_storage.get_session_memories(user_id, session_id)

                # 过滤 memory_type
                if memory_type:
                    memories = [m for m in memories if m.get("memory_type") == memory_type]

                all_memories.extend(memories)
        except Exception as e:
            logger.error(f"❌ 获取用户记忆失败: {e}")

        # 按重要性排序
        all_memories.sort(key=lambda x: x.get('importance', 1), reverse=True)

        return all_memories[:limit]

    def get_conversation_memories(self, conversation_id: int, user_id: int) -> List[Dict[str, Any]]:
        """获取特定会话的所有记忆"""
        session = self.session_storage.get_conversation_session(user_id, conversation_id)
        if not session:
            return []

        return self.session_storage.get_session_memories(user_id, session["id"])

    def delete(self, memory_id: int, user_id: int) -> bool:
        """删除记忆（从 ChromaDB 中删除，JSONL 不支持单条删除）"""
        chroma_service = ChromaService()
        chroma_service.delete_memory(user_id, memory_id)
        return True

    def clear_conversation_memories(self, conversation_id: int, user_id: int) -> int:
        """清空特定会话的记忆"""
        session = self.session_storage.get_conversation_session(user_id, conversation_id)
        if not session:
            return 0

        # 删除整个会话
        deleted = self.session_storage.delete_session(user_id, session["id"])

        # 同时从 ChromaDB 中删除相关记忆
        chroma_service = ChromaService()
        # 获取用户所有记忆并删除匹配的
        deleted_count = 0

        return deleted_count


class HybridRetrievalService:
    """混合检索服务 - 使用RRF融合向量和BM25检索，支持精排"""

    def __init__(self, session_storage: SessionStorage, chroma_service: ChromaService, user_id: int, alpha: float = 0.7):
        self.session_storage = session_storage
        self.chroma_service = chroma_service
        self.user_id = user_id
        self.alpha = alpha
        self.bm25 = BM25Service()
        self.rrf_fusion = RRFusion(k=60)

    def _initialize_bm25(self, memories: List[Dict[str, Any]]) -> None:
        """初始化BM25索引"""
        self.bm25.initialize(memories)

    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str = None,
        conversation_id: int = None,
        use_rerank: bool = True
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """使用RRF融合混合搜索，可选精排"""
        # 1. 从 SessionStorage 获取所有记忆用于BM25
        all_memories = self._get_all_memories(memory_type, conversation_id)

        if not all_memories:
            return []

        # 2. BM25检索
        self._initialize_bm25(all_memories)
        bm25_results = self.bm25.search(query, top_k * 3)

        # 3. 向量检索（从ChromaDB）
        chroma_results = self.chroma_service.search_memories(
            query=query,
            user_id=self.user_id,
            top_k=top_k * 3
        )

        # 4. RRF融合
        logger.info(f"🔄 Memory RRF融合开始，BM25: {len(bm25_results)}, 向量: {len(chroma_results)}")

        # 构建用于RRF融合的结果
        bm25_ranked = [(r['id'], i+1) for i, r in enumerate(bm25_results)]
        vector_ranked = []
        for i, r in enumerate(chroma_results):
            memory_id = r.get('id')
            if memory_id:
                vector_ranked.append((memory_id, i+1))

        # RRF融合
        rankings = [bm25_ranked, vector_ranked]
        fused_scores = self.rrf_fusion.fuse(rankings)

        # 5. 构建初步结果
        memories_dict = {m.get('id'): m for m in all_memories}
        memory_list = []
        
        for memory_id, rrf_score in fused_scores[:top_k * 2]:
            if memory_id in memories_dict:
                memory = memories_dict[memory_id].copy()
                memory['rrf_score'] = rrf_score
                memory_list.append(memory)

        # 6. 使用精排模型重新排序（可选）
        if use_rerank and query and memory_list:
            memory_list = embedding_service.rerank_with_metadata(query, memory_list)
            logger.info("🔄 使用精排模型重新排序完成")

        # 7. 确定来源并构建最终结果
        results = []
        for memory in memory_list[:top_k]:
            source = 'rerank' if use_rerank else 'rrf'
            if any(mid == memory.get('id') for mid, _ in bm25_ranked[:top_k]):
                source = 'bm25' if not use_rerank else 'rerank'
            elif any(mid == memory.get('id') for mid, _ in vector_ranked[:top_k]):
                source = 'chroma' if not use_rerank else 'rerank'

            results.append((memory, memory.get('relevance_score', memory.get('rrf_score', 0)), source))

        return results

    def _get_all_memories(self, memory_type: str = None, conversation_id: int = None) -> List[Dict[str, Any]]:
        """获取所有记忆"""
        all_memories = []

        index_path = self.session_storage._get_index_path(self.user_id)
        if not index_path.exists():
            return []

        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                sessions_data = json.load(f)

            for session_info in sessions_data.get("sessions", []):
                session_id = session_info["id"]

                # 按 conversation_id 过滤
                if conversation_id and session_info.get("conversation_id") != conversation_id:
                    continue

                memories = self.session_storage.get_session_memories(self.user_id, session_id)

                # 按 memory_type 过滤
                if memory_type:
                    memories = [m for m in memories if m.get("memory_type") == memory_type]

                all_memories.extend(memories)
        except Exception as e:
            logger.error(f"❌ 获取记忆失败: {e}")

        return all_memories

    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 1,
        conversation_id: Optional[int] = None,
        metadata: dict = None
    ) -> Dict[str, Any]:
        """添加记忆"""
        return LongTermMemory(self.session_storage).add(
            user_id=self.user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            conversation_id=conversation_id,
            metadata=metadata
        )


class MemoryService:
    """记忆服务主类 - 基于 SessionStorage + ChromaDB 双存储"""

    SHORT_TERM_MEMORY_MAX_SIZE = 10
    SHORT_TERM_MEMORY_TTL_SECONDS = 1800
    HYBRID_ALPHA = 0.7

    def __init__(self, user_id: int, conversation_id: Optional[int] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.chroma_service = ChromaService()
        self.session_storage = SessionStorage()

        self.short_term = ShortTermMemory(
            max_size=self.SHORT_TERM_MEMORY_MAX_SIZE,
            ttl_seconds=self.SHORT_TERM_MEMORY_TTL_SECONDS
        )
        self.long_term = LongTermMemory(self.session_storage)
        self.hybrid = HybridRetrievalService(
            self.session_storage,
            self.chroma_service,
            user_id,
            alpha=self.HYBRID_ALPHA
        )

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
        recent_messages = self.short_term.get_all()[-include_recent:]
        if recent_messages:
            context_parts.append("=== 最近对话 ===")
            for msg in recent_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                context_parts.append(f"{role}: {content}")

        # 长期记忆（混合检索）
        if self.conversation_id:
            memories = self.hybrid.search(
                query="",
                top_k=hybrid_top_k,
                conversation_id=self.conversation_id
            )
            if memories:
                context_parts.append("\n=== 相关记忆 ===")
                for memory, score, source in memories:
                    content = memory.get('content', '')[:200]
                    context_parts.append(f"[{source} {score:.3f}] {content}")

        return "\n".join(context_parts)

    def get_user_memories(self, memory_type: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户的所有记忆"""
        return self.long_term.get_user_memories(self.user_id, memory_type, limit)

    def get_conversation_memories(self) -> List[Dict[str, Any]]:
        """获取当前会话的所有记忆"""
        if not self.conversation_id:
            return []
        return self.long_term.get_conversation_memories(self.conversation_id, self.user_id)

    def clear_conversation_memories(self) -> int:
        """清空当前会话的记忆"""
        if not self.conversation_id:
            return 0
        return self.long_term.clear_conversation_memories(self.conversation_id, self.user_id)

    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 1,
        metadata: dict = None
    ) -> Dict[str, Any]:
        """添加记忆"""
        return self.hybrid.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            conversation_id=self.conversation_id,
            metadata=metadata
        )