from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range, TextIndexParams, TokenizerType
from qdrant_client.http.exceptions import UnexpectedResponse
from core.config import settings
from core.logger import logger
from typing import List, Optional, Dict, Any, Tuple
import uuid
import time


def retry_with_backoff(max_retries=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Qdrant operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))
            logger.error(f"Qdrant operation failed after {max_retries} attempts: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


class QdrantService:
    _instance = None
    _client: Optional[QdrantClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self) -> None:
        try:
            self._client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=30,
                check_compatibility=False
            )
            self._ensure_collection()
            logger.info(f"Qdrant connected: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self._client = None

    def _reconnect(self) -> bool:
        logger.info("Attempting to reconnect to Qdrant...")
        self._client = None
        self._connect()
        return self._client is not None

    def _ensure_collection(self) -> None:
        """确保所有必要的集合存在（BM25使用独立服务）"""
        if self._client is None:
            return
        try:
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            # 确保用户记忆集合存在
            if settings.QDRANT_COLLECTION not in collection_names:
                self._client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=settings.QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION}")
            
            # 确保RAG文档集合存在
            if settings.QDRANT_RAG_COLLECTION not in collection_names:
                self._client.create_collection(
                    collection_name=settings.QDRANT_RAG_COLLECTION,
                    vectors_config=VectorParams(
                        size=settings.QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant RAG collection: {settings.QDRANT_RAG_COLLECTION}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")

    def is_connected(self) -> bool:
        return self._client is not None

    @retry_with_backoff(max_retries=3, delay=1)
    def upsert_vector(
        self,
        user_id: int,
        memory_id: int,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> Optional[str]:
        if self._client is None:
            logger.warning("Qdrant not connected, attempting reconnect...")
            if not self._reconnect():
                logger.warning("Failed to reconnect to Qdrant, skipping upsert")
                return None
        try:
            point_id = memory_id
            self._client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "user_id": user_id,
                            "memory_id": memory_id,
                            **payload
                        }
                    )
                ]
            )
            logger.debug(f"Upserted vector: {point_id}")
            return point_id
        except Exception as e:
            logger.error(f"Failed to upsert vector: {e}")
            # 尝试重新连接
            try:
                self._reconnect()
            except Exception:
                pass
            return None

    @retry_with_backoff(max_retries=3, delay=1)
    def search_vectors(
        self,
        query_vector: List[float],
        user_id: int,
        limit: int = 5,
        score_threshold: float = 0.0,
        collection_name: str = None
    ) -> List[Dict[str, Any]]:
        if self._client is None:
            logger.warning("Qdrant not connected, attempting reconnect...")
            if not self._reconnect():
                logger.warning("Failed to reconnect to Qdrant, returning empty results")
                return []
        
        target_collection = collection_name or settings.QDRANT_COLLECTION
        
        try:
            # 兼容新旧版本的 Qdrant API
            if hasattr(self._client, 'search'):
                results = self._client.search(
                    collection_name=target_collection,
                    query_vector=query_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="user_id",
                                match=MatchValue(value=user_id)
                            )
                        ]
                    ),
                    limit=limit,
                    score_threshold=score_threshold
                )
            else:
                # 新版 QdrantClient 使用 query_points
                results = self._client.query_points(
                    collection_name=target_collection,
                    query=query_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="user_id",
                                match=MatchValue(value=user_id)
                            )
                        ]
                    ),
                    limit=limit,
                    score_threshold=score_threshold
                ).points
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            # 尝试重新连接
            try:
                self._reconnect()
            except Exception:
                pass
            return []

    def delete_vectors(self, memory_id: int, user_id: int) -> bool:
        if self._client is None:
            return False
        try:
            self._client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="memory_id", match=MatchValue(value=memory_id)),
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                )
            )
            logger.debug(f"Deleted vectors for memory_id={memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    def delete_user_vectors(self, user_id: int) -> bool:
        if self._client is None:
            return False
        try:
            self._client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                )
            )
            logger.info(f"Deleted all vectors for user_id={user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user vectors: {e}")
            return False

    @retry_with_backoff(max_retries=3, delay=1)
    def store_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        collection_name: str = None
    ) -> int:
        """批量存储向量"""
        if self._client is None:
            logger.warning("Qdrant not connected, attempting reconnect...")
            if not self._reconnect():
                logger.warning("Failed to reconnect to Qdrant, skipping store")
                return 0
        
        # 默认使用用户记忆集合，如果payload包含file_name则使用RAG集合
        target_collection = collection_name or settings.QDRANT_COLLECTION
        if not collection_name:
            for payload in payloads:
                if 'file_name' in payload:
                    target_collection = settings.QDRANT_RAG_COLLECTION
                    break
        
        try:
            points = []
            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                point_id = int(uuid.uuid4().hex[:15], 16)
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            self._client.upsert(
                collection_name=target_collection,
                points=points
            )
            logger.info(f"Stored {len(points)} vectors in collection: {target_collection}")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            try:
                self._reconnect()
            except Exception:
                pass
            return 0

    def upsert_memory(
        self,
        user_id: int,
        memory_id: str,
        content: str,
        vector: List[float],
        memory_type: str = "general",
        importance: int = 1,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """存储记忆向量到Qdrant"""
        if self._client is None:
            if not self._reconnect():
                return False
        
        try:
            payload = {
                "user_id": user_id,
                "memory_id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "metadata": metadata or {}
            }
            
            self._client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=[PointStruct(
                    id=int(memory_id) if memory_id.isdigit() else int(uuid.uuid4().hex[:15], 16),
                    vector=vector,
                    payload=payload
                )]
            )
            logger.debug(f"Upserted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert memory: {e}")
            return False

    @retry_with_backoff(max_retries=3, delay=1)
    def search_vectors_for_rag(
        self,
        query_vector: List[float],
        user_id: int,
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """为RAG搜索向量 - 使用RAG专用集合"""
        if self._client is None:
            logger.warning("Qdrant not connected, attempting reconnect...")
            if not self._reconnect():
                logger.warning("Failed to reconnect to Qdrant, returning empty results")
                return []
        try:
            if hasattr(self._client, 'search'):
                results = self._client.search(
                    collection_name=settings.QDRANT_RAG_COLLECTION,
                    query_vector=query_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="user_id",
                                match=MatchValue(value=user_id)
                            )
                        ]
                    ),
                    limit=limit,
                    score_threshold=score_threshold
                )
            else:
                results = self._client.query_points(
                    collection_name=settings.QDRANT_RAG_COLLECTION,
                    query=query_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="user_id",
                                match=MatchValue(value=user_id)
                            )
                        ]
                    ),
                    limit=limit,
                    score_threshold=score_threshold
                ).points
            
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "content": r.payload.get("content", ""),
                    "file_name": r.payload.get("file_name", ""),
                    "file_path": r.payload.get("file_path", ""),
                    "chunk_index": r.payload.get("chunk_index", 0)
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to search vectors for RAG: {e}")
            try:
                self._reconnect()
            except Exception:
                pass
            return []

    def delete_vectors_by_file(self, file_path: str, user_id: int) -> bool:
        """根据文件路径删除向量 - 从RAG集合中删除"""
        if self._client is None:
            return False
        try:
            self._client.delete(
                collection_name=settings.QDRANT_RAG_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                )
            )
            logger.info(f"Deleted vectors for file_path={file_path} from RAG collection")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors by file: {e}")
            return False

    def count_vectors(self, user_id: int, collection_name: str = None) -> int:
        """统计用户的向量数量"""
        if self._client is None:
            return 0
        try:
            target_collection = collection_name or settings.QDRANT_COLLECTION
            result = self._client.count(
                collection_name=target_collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                )
            )
            return result.count
        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0

    def count_rag_vectors(self, user_id: int) -> int:
        """统计用户RAG向量的数量"""
        return self.count_vectors(user_id, settings.QDRANT_RAG_COLLECTION)

    def list_user_files(self, user_id: int) -> List[Dict[str, Any]]:
        """获取用户的所有文件列表（基于RAG集合中的file_path去重）"""
        if self._client is None:
            return []
        try:
            # 使用scroll获取所有属于该用户的记录
            results = self._client.scroll(
                collection_name=settings.QDRANT_RAG_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                ),
                limit=10000,
                with_payload=True
            )
            
            # 按file_path分组，统计chunk数量
            file_info: Dict[str, Dict[str, Any]] = {}
            for point in results[0]:
                payload = point.payload
                file_path = payload.get("file_path", "")
                if file_path:
                    if file_path not in file_info:
                        file_info[file_path] = {
                            "file_name": payload.get("file_name", ""),
                            "file_path": file_path,
                            "chunk_count": 0,
                            "created_at": payload.get("created_at", "")
                        }
                    file_info[file_path]["chunk_count"] += 1
            
            return list(file_info.values())
        except Exception as e:
            logger.error(f"Failed to list user files: {e}")
            return []

    def get_all_rag_contents_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        """获取用户所有RAG内容用于BM25检索"""
        if self._client is None:
            return []
        try:
            results = self._client.scroll(
                collection_name=settings.QDRANT_RAG_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                ),
                limit=100000,
                with_payload=True
            )
            
            return [
                {
                    "id": point.id,
                    "content": point.payload.get("content", ""),
                    "file_name": point.payload.get("file_name", ""),
                    "file_path": point.payload.get("file_path", ""),
                    "chunk_index": point.payload.get("chunk_index", 0)
                }
                for point in results[0]
            ]
        except Exception as e:
            logger.error(f"Failed to get all RAG contents: {e}")
            return []

    def get_all_memory_contents_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        """获取用户所有记忆内容用于BM25检索"""
        if self._client is None:
            return []
        try:
            results = self._client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                ),
                limit=100000,
                with_payload=True
            )
            
            return [
                {
                    "id": point.payload.get("memory_id", point.id),
                    "content": point.payload.get("content", ""),
                    "memory_type": point.payload.get("memory_type", "general"),
                    "importance": point.payload.get("importance", 1),
                    "conversation_id": point.payload.get("conversation_id")
                }
                for point in results[0]
            ]
        except Exception as e:
            logger.error(f"Failed to get all memory contents: {e}")
            return []

    def delete_all_user_rag(self, user_id: int) -> bool:
        """删除用户的所有RAG数据"""
        if self._client is None:
            return False
        try:
            self._client.delete(
                collection_name=settings.QDRANT_RAG_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                )
            )
            logger.info(f"Deleted all RAG vectors for user_id={user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all user RAG: {e}")
            return False

    def delete_all_user_memories(self, user_id: int) -> bool:
        """删除用户的所有记忆数据"""
        if self._client is None:
            return False
        try:
            self._client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                )
            )
            logger.info(f"Deleted all memory vectors for user_id={user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all user memories: {e}")
            return False


qdrant_service = QdrantService()
