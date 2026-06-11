"""
BM25检索服务 - 独立模块，支持中文和英文
"""
from typing import List, Dict, Any, Tuple
from collections import Counter
import math
import re
import jieba
from core.logger import logger


class BM25Service:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_lens: List[int] = []
        self.idf: Dict[str, float] = {}
        self.doc_count_per_term: Dict[str, int] = {}
        self.avg_doc_len: float = 0
        self._jieba_available = False
        self._init_jieba()
        
    def _init_jieba(self) -> None:
        """尝试初始化 jieba 分词器"""
        try:
            import jieba
            self._jieba = jieba
            self._jieba_available = True
            logger.info("✅ BM25 jieba 分词器已加载")
        except ImportError:
            self._jieba_available = False
            logger.warning("⚠️ BM25 jieba 未安装，将使用字符级分词处理中文")
    
    def _contains_chinese(self, text: str) -> bool:
        """检测文本是否包含中文"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
    
    def tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        # 处理输入可能是列表的情况
        if isinstance(text, list):
            # 如果是列表，将所有元素拼接成字符串
            text = " ".join(str(t) for t in text)
        
        text = text.lower()
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            '我', '你', '他', '她', '它', '我们', '你们', '他们', '的', '了', '在',
            '是', '有', '和', '就', '不', '人', '都', '一', '一个', '上',
            '也', '很', '到', '说', '要', '去', '会', '着', '没有', '看',
            '好', '自己', '这', '那', '么', '什么', '怎么', '为什么',
            '个', '把', '被', '让', '给', '跟', '从', '向', '对', '过',
            '呢', '啊', '吧', '吗', '嘛', '哦', '呀', '哈', '嘿'
        }
        
        if self._contains_chinese(text):
            # 中文文本使用 jieba 分词
            if self._jieba_available:
                tokens = self._jieba.lcut(text)
            else:
                # 降级为字符级分词
                tokens = list(text)
        else:
            # 纯英文文本使用正则分词
            tokens = re.findall(r'\b\w+\b', text)
        
        return [t for t in tokens if t not in stop_words and len(t) > 1]
    
    def initialize(self, documents: List[Dict[str, Any]]) -> None:
        """初始化BM25索引"""
        if not documents:
            logger.warning("⚠️ BM25初始化：文档列表为空")
            return
            
        self.documents = documents
        self.doc_term_freqs = []
        self.doc_lens = []
        self.idf = {}
        self.doc_count_per_term = {}
        
        doc_count = len(documents)
        
        for doc in documents:
            content = doc.get('content', '')
            tokens = self.tokenize(content)
            tf = Counter(tokens)
            self.doc_term_freqs.append(dict(tf))
            self.doc_lens.append(len(tokens))
            
            # 统计词项文档频率
            for term in tf.keys():
                self.doc_count_per_term[term] = self.doc_count_per_term.get(term, 0) + 1
        
        self.avg_doc_len = sum(self.doc_lens) / doc_count if doc_count > 0 else 0
        
        # 计算IDF
        for term, term_doc_count in self.doc_count_per_term.items():
            self.idf[term] = math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
        
        logger.info(f"✅ BM25索引初始化完成，文档数: {doc_count}, 词项数: {len(self.idf)}")
    
    def _score_single(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单个文档的BM25分数"""
        if doc_idx >= len(self.doc_term_freqs):
            return 0.0
            
        doc_tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        score = 0.0
        
        for term in query_tokens:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self.idf.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len + 1e-8))
            score += idf * (numerator / denominator) if denominator > 0 else 0
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索最相关的文档"""
        if not self.documents:
            return []
        
        query_tokens = self.tokenize(query)
        
        if not query_tokens:
            return []
        
        # 计算所有文档的分数
        scores = []
        for i, doc in enumerate(self.documents):
            score = self._score_single(query_tokens, i)
            if score > 0:
                scores.append({
                    'id': doc.get('id', i),
                    'score': score,
                    'content': doc.get('content', ''),
                    'file_name': doc.get('file_name', ''),
                    'file_path': doc.get('file_path', '')
                })
        
        # 按分数排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top_k
        return scores[:top_k]
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """批量搜索"""
        return [self.search(query, top_k) for query in queries]


class BM25FuzzySearch:
    """带模糊匹配的BM25搜索"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25Service(k1, b)
    
    def initialize(self, documents: List[Dict[str, Any]]) -> None:
        """初始化索引"""
        self.bm25.initialize(documents)
    
    def search_with_fuzzy(self, query: str, top_k: int = 10, fuzzy_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """带模糊匹配的搜索"""
        # 先进行精确匹配
        exact_results = self.bm25.search(query, top_k * 2)
        
        if len(exact_results) >= top_k:
            return exact_results[:top_k]
        
        # 如果精确匹配不足，尝试模糊匹配
        fuzzy_results = self._fuzzy_search(query, top_k * 2)
        
        # 合并结果，精确匹配优先
        seen_ids = {r['id'] for r in exact_results}
        merged = list(exact_results)
        
        for result in fuzzy_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                merged.append(result)
        
        return merged[:top_k]
    
    def _fuzzy_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """简单的模糊搜索"""
        if not self.bm25.documents:
            return []
        
        query_lower = query.lower()
        query_terms = set(self.bm25.tokenize(query))
        
        results = []
        for i, doc in enumerate(self.bm25.documents):
            content_lower = doc.get('content', '').lower()
            doc_terms = set(self.bm25.tokenize(doc.get('content', '')))
            
            # 计算词项重叠度
            if query_terms and doc_terms:
                overlap = len(query_terms & doc_terms) / len(query_terms)
                if overlap >= 0.3:
                    results.append({
                        'id': doc.get('id', i),
                        'score': overlap,
                        'content': doc.get('content', ''),
                        'file_name': doc.get('file_name', ''),
                        'file_path': doc.get('file_path', '')
                    })
            
            # 检查是否有部分匹配
            if query_lower in content_lower:
                results.append({
                    'id': doc.get('id', i),
                    'score': 1.0,
                    'content': doc.get('content', ''),
                    'file_name': doc.get('file_name', ''),
                    'file_path': doc.get('file_path', '')
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
