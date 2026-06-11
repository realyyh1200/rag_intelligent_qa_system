"""ChromaDB 向量数据库服务 - 使用轻量级本地嵌入"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
from core.config import settings
from core.logger import logger
from services.embedding_service import embedding_service, SimpleEmbedding
import numpy as np
import time


class CustomEmbeddingFunction:
    """自定义嵌入函数 - 使用轻量级本地嵌入（不下载模型）"""
    
    def __init__(self):
        self.simple_embedding = SimpleEmbedding(dimension=384)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """生成文本嵌入向量 - ChromaDB 0.4.16+ 使用 input 作为参数名"""
        return self.simple_embedding.encode(input)
    
    def embed_query(self, input) -> List[float]:
        """生成单个查询文本的嵌入向量 - ChromaDB 使用 input 作为参数名"""
        # ChromaDB 可能传递字符串或列表
        if isinstance(input, str):
            result = self.simple_embedding.encode([input])
            return result[0] if result else [0.0] * 384
        elif isinstance(input, list):
            # 如果是列表，取第一个元素处理
            if len(input) > 0 and isinstance(input[0], str):
                result = self.simple_embedding.encode(input)
                return result[0] if result else [0.0] * 384
            else:
                # 空列表或嵌套列表，返回默认向量
                return [0.0] * 384
        else:
            return [0.0] * 384
    
    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """生成多个文档的嵌入向量 - ChromaDB 使用 input 作为参数名"""
        return self.simple_embedding.encode(input)
    
    def name(self) -> str:
        """返回嵌入函数名称"""
        return "simple_local_embedding"


class ChromaService:
    """ChromaDB 向量数据库服务"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                # 增加超时设置
                request_timeout=60.0,
                chroma_server_http_port=8000
            )
        )
        
        # 使用自定义嵌入函数（基于 sentence-transformers）
        self.embedding_function = CustomEmbeddingFunction()
        
        self._initialized = True
        logger.info("✅ ChromaDB 服务初始化完成 (使用轻量级本地嵌入)")
    
    def _safe_upsert(self, collection, ids, documents, metadatas, operation_name: str = "upsert"):
        """安全的 upsert 操作，带重试机制"""
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                return
            except Exception as e:
                if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ {operation_name} 超时，第 {attempt + 1}/{max_retries} 次重试...")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"❌ {operation_name} 超时，已达到最大重试次数")
                raise e
    
    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """获取或创建集合"""
        try:
            # 获取现有集合，同时指定嵌入函数
            return self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            # 如果集合不存在，创建新集合并使用自定义嵌入函数
            return self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
    
    def upsert_memory(self,
                     user_id: int,
                     memory_id: int,
                     content: str,
                     memory_type: str = "general",
                     importance: int = 1,
                     conversation_id: Optional[int] = None,
                     metadata: Dict[str, Any] = None) -> None:
        """插入或更新记忆向量"""
        collection = self.get_or_create_collection(settings.CHROMA_MEMORY_COLLECTION)
        
        doc_id = f"user_{user_id}_memory_{memory_id}"
        
        payload = {
            'user_id': user_id,
            'memory_id': memory_id,
            'memory_type': memory_type,
            'content': content,
            'importance': importance,
            'conversation_id': conversation_id,
            'created_at': metadata.get('created_at') if metadata else None,
        }
        # 将额外的 metadata 扁平化为字符串存储
        if metadata and isinstance(metadata, dict):
            for key, value in metadata.items():
                if key not in payload and isinstance(value, (str, int, float, bool, type(None))):
                    payload[f"meta_{key}"] = value
        
        self._safe_upsert(
            collection=collection,
            ids=[doc_id],
            documents=[content],
            metadatas=[payload],
            operation_name=f"记忆 upsert ({doc_id})"
        )
        
        logger.debug(f"✅ 记忆已存入 ChromaDB: {doc_id}")
    
    def _get_query_embedding(self, query: str) -> List[List[float]]:
        """获取查询的嵌入向量"""
        return self.embedding_function([query])
    
    def search_memories(self,
                       query: str,
                       user_id: int,
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关记忆"""
        collection = self.get_or_create_collection(settings.CHROMA_MEMORY_COLLECTION)
        
        # 手动生成嵌入向量
        query_embeddings = self._get_query_embedding(query)
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where={
                "user_id": user_id
            }
        )
        
        memories = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            memories.append({
                'id': metadata.get('memory_id'),
                'content': doc,
                'memory_type': metadata.get('memory_type'),
                'importance': metadata.get('importance', 1),
                'conversation_id': metadata.get('conversation_id'),
                'relevance_score': float(1 - distance),  # 转换为相似度分数
            })
        
        return memories
    
    def search_memories_with_rerank(self,
                                   query: str,
                                   user_id: int,
                                   top_k: int = 5,
                                   rerank_top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相关记忆并使用精排模型重新排序"""
        # 先获取较多候选结果
        collection = self.get_or_create_collection(settings.CHROMA_MEMORY_COLLECTION)
        
        # 手动生成嵌入向量
        query_embeddings = self._get_query_embedding(query)
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=rerank_top_k,
            where={
                "user_id": user_id
            }
        )
        
        # 构建候选列表
        candidates = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            candidates.append({
                'id': metadata.get('memory_id'),
                'content': doc,
                'memory_type': metadata.get('memory_type'),
                'importance': metadata.get('importance', 1),
                'conversation_id': metadata.get('conversation_id'),
            })
        
        # 使用精排模型重新排序
        reranked = embedding_service.rerank_with_metadata(query, candidates)
        
        return reranked[:top_k]
    
    def delete_memory(self, user_id: int, memory_id: int) -> None:
        """删除记忆"""
        collection = self.get_or_create_collection(settings.CHROMA_MEMORY_COLLECTION)
        
        doc_id = f"user_{user_id}_memory_{memory_id}"
        
        try:
            collection.delete(ids=[doc_id])
            logger.debug(f"✅ 记忆已从 ChromaDB 删除: {doc_id}")
        except Exception as e:
            logger.warning(f"⚠️ 删除记忆失败: {e}")
    
    def upsert_rag_document(self,
                           user_id: int,
                           doc_id: str,
                           content: str,
                           file_name: str = "",
                           metadata: Dict[str, Any] = None) -> None:
        """插入或更新 RAG 文档向量"""
        collection = self.get_or_create_collection(settings.CHROMA_RAG_COLLECTION)
        
        payload = {
            'user_id': user_id,
            'doc_id': doc_id,
            'file_name': file_name,
            'content': content,
        }
        # 将额外的 metadata 扁平化为字符串存储
        if metadata and isinstance(metadata, dict):
            for key, value in metadata.items():
                if key not in payload and isinstance(value, (str, int, float, bool, type(None))):
                    payload[f"meta_{key}"] = value
        
        self._safe_upsert(
            collection=collection,
            ids=[f"user_{user_id}_doc_{doc_id}"],
            documents=[content],
            metadatas=[payload],
            operation_name=f"RAG 文档 upsert ({doc_id})"
        )
        
        logger.debug(f"✅ RAG 文档已存入 ChromaDB: {doc_id}")
    
    def search_rag_documents(self,
                            query: str,
                            user_id: int,
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关 RAG 文档"""
        collection = self.get_or_create_collection(settings.CHROMA_RAG_COLLECTION)
        
        # 手动生成嵌入向量
        query_embeddings = self._get_query_embedding(query)
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where={
                "user_id": user_id
            }
        )
        
        documents = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            documents.append({
                'doc_id': metadata.get('doc_id'),
                'content': doc,
                'file_name': metadata.get('file_name', ''),
                'relevance_score': float(1 - distance),
            })
        
        return documents
    
    def search_rag_documents_with_rerank(self,
                                        query: str,
                                        user_id: int,
                                        top_k: int = 5,
                                        rerank_top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索 RAG 文档并使用精排模型重新排序"""
        # 先获取较多候选结果
        collection = self.get_or_create_collection(settings.CHROMA_RAG_COLLECTION)
        
        # 手动生成嵌入向量
        query_embeddings = self._get_query_embedding(query)
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=rerank_top_k,
            where={
                "user_id": user_id
            }
        )
        
        # 构建候选列表
        candidates = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            candidates.append({
                'doc_id': metadata.get('doc_id'),
                'content': doc,
                'file_name': metadata.get('file_name', ''),
            })
        
        # 使用精排模型重新排序
        reranked = embedding_service.rerank_with_metadata(query, candidates)
        
        return reranked[:top_k]
    
    def delete_rag_document(self, user_id: int, doc_id: str) -> None:
        """删除 RAG 文档"""
        collection = self.get_or_create_collection(settings.CHROMA_RAG_COLLECTION)
        
        try:
            collection.delete(ids=[f"user_{user_id}_doc_{doc_id}"])
            logger.debug(f"✅ RAG 文档已从 ChromaDB 删除: {doc_id}")
        except Exception as e:
            logger.warning(f"⚠️ 删除 RAG 文档失败: {e}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        collection = self.get_or_create_collection(collection_name)
        return {
            'count': collection.count(),
            'name': collection_name
        }


# 创建单例实例
chroma_service = ChromaService()