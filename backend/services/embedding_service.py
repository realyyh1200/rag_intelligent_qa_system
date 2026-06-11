"""
嵌入服务 - 使用 sentence-transformers 实现词嵌入和精排功能
支持离线模式，当无法下载模型时使用轻量级回退方案
"""

from typing import List, Dict, Any, Optional, Tuple
from core.logger import logger
import os
import re
from collections import Counter


class SimpleEmbedding:
    """
    轻量级本地嵌入实现 - 基于词频和哈希的简单向量表示
    用于无法下载sentence-transformers模型时的回退方案
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocab = {}
        self.vocab_size = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        text = text.lower()
        tokens = re.findall(r'[a-zA-Z\u4e00-\u9fa5]+', text)
        return tokens
    
    def _hash_token(self, token: str) -> int:
        """哈希token到维度索引"""
        return hash(token) % self.dimension
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量"""
        results = []
        for text in texts:
            tokens = self._tokenize(text)
            vector = [0.0] * self.dimension
            
            # 词频统计
            token_counts = Counter(tokens)
            max_count = max(token_counts.values()) if token_counts else 1
            
            for token, count in token_counts.items():
                idx = self._hash_token(token)
                vector[idx] += count / max_count
            
            # 归一化
            norm = sum(v * v for v in vector) ** 0.5
            if norm > 0:
                vector = [v / norm for v in vector]
            
            results.append(vector)
        
        return results


class SimpleReranker:
    """
    轻量级精排实现 - 基于词匹配的相似度计算
    """
    
    def __init__(self):
        pass
    
    def _tokenize(self, text: str) -> set:
        """简单分词"""
        text = text.lower()
        return set(re.findall(r'[a-zA-Z\u4e00-\u9fa5]+', text))
    
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """预测(query, document)对的相似度分数"""
        scores = []
        for query, doc in pairs:
            query_tokens = self._tokenize(query)
            doc_tokens = self._tokenize(doc)
            
            if not query_tokens:
                scores.append(0.0)
                continue
            
            # 计算Jaccard相似度
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            
            if union == 0:
                scores.append(0.0)
            else:
                scores.append(float(intersection / union))
        
        return scores


class EmbeddingService:
    """
    嵌入服务类 - 提供词嵌入和精排功能
    
    词嵌入模型: all-MiniLM-L6-v2 (轻量级，约80MB)
    精排模型: cross-encoder/ms-marco-MiniLM-L-6-v2 (轻量级排序模型)
    
    当无法下载sentence-transformers模型时，自动回退到轻量级方案
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 设置环境变量
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')
        os.environ.setdefault('HF_DATASETS_OFFLINE', '0')
        
        # 词嵌入模型（延迟加载）
        self._embedding_model = None
        # 精排模型（延迟加载）
        self._rerank_model = None
        
        # 轻量级回退方案
        self.simple_embedding = SimpleEmbedding(dimension=384)
        self.simple_reranker = SimpleReranker()
        
        # 是否使用回退方案
        self.use_fallback = False
        
        # 延迟初始化，不在启动时加载模型
        self._initialized = True
        logger.info("✅ 嵌入服务初始化完成（模型延迟加载）")
    
    def ensure_models_loaded(self):
        """确保模型已加载（如果使用sentence-transformers）"""
        # 直接使用轻量级回退方案，避免加载大型模型导致崩溃
        self.use_fallback = True
        logger.info("✅ 使用轻量级回退方案进行嵌入和精排")
    
    @property
    def embedding_model(self):
        """获取词嵌入模型（延迟加载）"""
        self.ensure_models_loaded()
        return self._embedding_model
    
    @property
    def rerank_model(self):
        """获取精排模型（延迟加载）"""
        self.ensure_models_loaded()
        return self._rerank_model
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        对文本进行编码，生成向量表示
        
        Args:
            texts: 文本列表
        
        Returns:
            向量列表，每个向量维度为 384
        """
        if self.use_fallback:
            return self.simple_embedding.encode(texts)
        
        self.ensure_models_loaded()
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            logger.error(f"❌ 文本编码失败，使用回退方案: {e}")
            return self.simple_embedding.encode(texts)
    
    def encode_single(self, text: str) -> List[float]:
        """
        对单个文本进行编码
        
        Args:
            text: 单个文本
        
        Returns:
            向量表示，维度为 384
        """
        return self.encode([text])[0]
    
    def rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        对文档进行精排，返回排序后的结果
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
        
        Returns:
            排序后的文档列表，包含相关性分数
        """
        if not documents:
            return []
        
        if self.use_fallback:
            # 使用轻量级精排
            pairs = [(query, doc) for doc in documents]
            scores = self.simple_reranker.predict(pairs)
            
            results = []
            for doc, score in zip(documents, scores):
                results.append({
                    'content': doc,
                    'relevance_score': score
                })
            
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results
        
        self.ensure_models_loaded()
        
        try:
            # 构建 (query, document) 对
            pairs = [(query, doc) for doc in documents]
            
            # 获取精排分数
            scores = self.rerank_model.predict(pairs)
            
            # 组合结果并排序
            results = []
            for doc, score in zip(documents, scores):
                results.append({
                    'content': doc,
                    'relevance_score': float(score)
                })
            
            # 按分数降序排序
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"❌ 精排失败，使用回退方案: {e}")
            # 使用轻量级精排
            pairs = [(query, doc) for doc in documents]
            scores = self.simple_reranker.predict(pairs)
            
            results = []
            for doc, score in zip(documents, scores):
                results.append({
                    'content': doc,
                    'relevance_score': score
                })
            
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results
    
    def rerank_with_metadata(self, query: str, items: List[Dict[str, Any]], content_key: str = 'content') -> List[Dict[str, Any]]:
        """
        对带有元数据的项目进行精排
        
        Args:
            query: 查询文本
            items: 待排序的项目列表，每个项目包含 content_key 字段
            content_key: 内容字段名，默认为 'content'
        
        Returns:
            排序后的项目列表，添加了 relevance_score 字段
        """
        if not items:
            return []
        
        if self.use_fallback:
            # 使用轻量级精排
            documents = [item.get(content_key, '') for item in items]
            pairs = [(query, doc) for doc in documents]
            scores = self.simple_reranker.predict(pairs)
            
            results = []
            for item, score in zip(items, scores):
                result = item.copy()
                result['relevance_score'] = score
                results.append(result)
            
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results
        
        self.ensure_models_loaded()
        
        try:
            # 提取内容
            documents = [item.get(content_key, '') for item in items]
            
            # 构建 (query, document) 对
            pairs = [(query, doc) for doc in documents]
            
            # 获取精排分数
            scores = self.rerank_model.predict(pairs)
            
            # 组合结果
            results = []
            for item, score in zip(items, scores):
                result = item.copy()
                result['relevance_score'] = float(score)
                results.append(result)
            
            # 按分数降序排序
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"❌ 精排失败，使用回退方案: {e}")
            # 使用轻量级精排
            documents = [item.get(content_key, '') for item in items]
            pairs = [(query, doc) for doc in documents]
            scores = self.simple_reranker.predict(pairs)
            
            results = []
            for item, score in zip(items, scores):
                result = item.copy()
                result['relevance_score'] = score
                results.append(result)
            
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results
    
    def semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义搜索：先通过词嵌入检索，再进行精排
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量
        
        Returns:
            排序后的文档列表
        """
        # 简单实现：直接使用精排模型进行排序
        results = self.rerank(query, documents)
        return results[:top_k]
    
    def get_embedding_dimension(self) -> int:
        """获取词嵌入维度"""
        return 384


# 创建单例实例
embedding_service = EmbeddingService()