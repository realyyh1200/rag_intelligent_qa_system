"""
RRF (Reciprocal Rank Fusion) 融合工具
一种鲁棒的排序融合方法，比简单加权求和更稳定
"""
from typing import List, Dict, Any, Tuple
from core.logger import logger


class RRFusion:
    """
    倒数排名融合（Reciprocal Rank Fusion）
    
    RRF公式: RRF_score(d) = Σ 1/(k + rank(d))
    
    优点：
    1. 对单个检索器的分数不敏感
    2. 不需要调参（k值固定为60）
    3. 对排名顺序更敏感，适合融合多个检索器
    4. 避免分数归一化的问题
    """
    
    def __init__(self, k: int = 60):
        """
        初始化RRF融合器
        
        Args:
            k: 排名惩罚参数，值越大，后续排名的权重下降越慢
               常用值: 60 (默认值), 10-100之间调整
        """
        self.k = k
    
    def fuse(
        self,
        rankings: List[List[Tuple[Any, int]]],
        scores: List[List[float]] = None
    ) -> List[Tuple[Any, float]]:
        """
        融合多个排名列表
        
        Args:
            rankings: 排名列表的列表，每个元素是 (item_id, rank) 的列表
                     rank从1开始，1表示第一名
            scores: 可选的分数列表，用于辅助排序（在RRF分数相同时）
                   如果不提供，则按RRF分数降序排列
        
        Returns:
            融合后的排序列表，每个元素是 (item_id, rrf_score)
            按RRF分数降序排列
        """
        if not rankings:
            return []
        
        # 计算每个item的RRF分数
        rrf_scores: Dict[Any, float] = {}
        
        for ranking in rankings:
            for rank, (item_id, _) in enumerate(ranking, start=1):
                # RRF公式: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                rrf_scores[item_id] = rrf_scores.get(item_id, 0) + rrf_score
        
        # 排序
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items
    
    def fuse_with_scores(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]] = None,
        id_field: str = 'id'
    ) -> List[Dict[str, Any]]:
        """
        融合向量检索、BM25检索和关键词检索的结果
        
        Args:
            vector_results: 向量检索结果列表
            bm25_results: BM25检索结果列表  
            keyword_results: 关键词检索结果列表（可选）
            id_field: 用于标识文档的字段名
        
        Returns:
            融合后的结果列表，包含所有分数信息
        """
        # 构建排名列表
        rankings = []
        
        # 向量检索排名
        if vector_results:
            vector_ranking = [(r.get(id_field, r.get('content', '')), i+1) 
                            for i, r in enumerate(vector_results)]
            rankings.append(vector_ranking)
        
        # BM25排名
        if bm25_results:
            bm25_ranking = [(r.get(id_field, r.get('content', '')), i+1) 
                           for i, r in enumerate(bm25_results)]
            rankings.append(bm25_ranking)
        
        # 关键词排名（如果有）
        if keyword_results:
            keyword_ranking = [(r.get(id_field, r.get('content', '')), i+1) 
                              for i, r in enumerate(keyword_results)]
            rankings.append(keyword_ranking)
        
        if not rankings:
            return []
        
        # 执行RRF融合
        fused_scores = self.fuse(rankings)
        
        # 构建结果字典
        result_dict = {}
        
        # 添加向量检索结果
        for i, r in enumerate(vector_results):
            item_id = r.get(id_field, r.get('content', ''))
            if item_id not in result_dict:
                result_dict[item_id] = {
                    **r,
                    'vector_rank': i + 1,
                    'rrf_score': 0.0
                }
        
        # 添加BM25结果
        for i, r in enumerate(bm25_results):
            item_id = r.get(id_field, r.get('content', ''))
            if item_id not in result_dict:
                result_dict[item_id] = {
                    **r,
                    'bm25_rank': i + 1,
                    'rrf_score': 0.0
                }
            else:
                result_dict[item_id]['bm25_rank'] = i + 1
        
        # 添加关键词结果
        if keyword_results:
            for i, r in enumerate(keyword_results):
                item_id = r.get(id_field, r.get('content', ''))
                if item_id in result_dict:
                    result_dict[item_id]['keyword_rank'] = i + 1
        
        # 更新RRF分数
        for item_id, rrf_score in fused_scores:
            if item_id in result_dict:
                result_dict[item_id]['rrf_score'] = rrf_score
        
        # 转换为列表并按RRF分数排序
        fused_results = list(result_dict.values())
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return fused_results


def simple_rrf_fusion(
    results_lists: List[List[Dict[str, Any]]],
    id_field: str = 'id',
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    简单的RRF融合函数
    
    Args:
        results_lists: 多个检索结果列表
        id_field: 文档标识字段
        k: RRF参数
    
    Returns:
        融合后的结果列表
    """
    fusion = RRFusion(k=k)
    
    # 构建每个结果的排名
    rankings = []
    all_items = {}
    
    for results in results_lists:
        ranking = []
        for i, item in enumerate(results):
            item_id = item.get(id_field, str(i))
            ranking.append((item_id, i + 1))
            if item_id not in all_items:
                all_items[item_id] = item
        rankings.append(ranking)
    
    # 执行融合
    fused_scores = fusion.fuse(rankings)
    
    # 构建结果
    results = []
    for item_id, rrf_score in fused_scores:
        if item_id in all_items:
            result = {**all_items[item_id], 'rrf_score': rrf_score}
            results.append(result)
    
    return results
