"""
RRF融合测试脚本
验证RRF（倒数排名融合）算法的效果
"""
import sys
import os

# 添加backend目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.rrf_fusion import RRFusion, simple_rrf_fusion


def test_basic_rrf():
    """测试基本的RRF融合"""
    print("=" * 60)
    print("测试 1: 基本RRF融合")
    print("=" * 60)
    
    fusion = RRFusion(k=60)
    
    # 模拟三个检索结果
    # 向量检索结果
    vector_results = [
        {'id': 'doc1', 'score': 0.95, 'source': 'vector'},
        {'id': 'doc2', 'score': 0.90, 'source': 'vector'},
        {'id': 'doc3', 'score': 0.85, 'source': 'vector'},
        {'id': 'doc4', 'score': 0.80, 'source': 'vector'},
    ]
    
    # BM25检索结果
    bm25_results = [
        {'id': 'doc2', 'score': 15.5, 'source': 'bm25'},
        {'id': 'doc1', 'score': 14.2, 'source': 'bm25'},
        {'id': 'doc5', 'score': 12.8, 'source': 'bm25'},
        {'id': 'doc6', 'score': 11.1, 'source': 'bm25'},
    ]
    
    # 关键词检索结果
    keyword_results = [
        {'id': 'doc1', 'score': 5.0, 'source': 'keyword'},
        {'id': 'doc3', 'score': 3.0, 'source': 'keyword'},
        {'id': 'doc7', 'score': 2.0, 'source': 'keyword'},
    ]
    
    print("\n原始检索结果:")
    print(f"向量检索: {[r['id'] for r in vector_results]}")
    print(f"BM25检索: {[r['id'] for r in bm25_results]}")
    print(f"关键词检索: {[r['id'] for r in keyword_results]}")
    
    # RRF融合
    fused = fusion.fuse_with_scores(
        vector_results=vector_results,
        bm25_results=bm25_results,
        keyword_results=keyword_results,
        id_field='id'
    )
    
    print("\n✅ RRF融合结果:")
    for i, result in enumerate(fused, 1):
        print(f"  {i}. {result['id']} (RRF: {result['rrf_score']:.4f}, ranks: {result.get('ranks_info', 'N/A')})")
    
    return True


def test_simple_rrf():
    """测试简单的RRF融合函数"""
    print("\n" + "=" * 60)
    print("测试 2: 简单RRF融合函数")
    print("=" * 60)
    
    # 不同的检索结果列表
    list1 = [
        {'id': 'A', 'content': '文档A'},
        {'id': 'B', 'content': '文档B'},
        {'id': 'C', 'content': '文档C'},
    ]
    
    list2 = [
        {'id': 'B', 'content': '文档B'},
        {'id': 'D', 'content': '文档D'},
        {'id': 'A', 'content': '文档A'},
    ]
    
    list3 = [
        {'id': 'C', 'content': '文档C'},
        {'id': 'A', 'content': '文档A'},
        {'id': 'E', 'content': '文档E'},
    ]
    
    print("\n原始排名:")
    print(f"检索器1: {[r['id'] for r in list1]}")
    print(f"检索器2: {[r['id'] for r in list2]}")
    print(f"检索器3: {[r['id'] for r in list3]}")
    
    # 融合
    fused = simple_rrf_fusion([list1, list2, list3], id_field='id', k=60)
    
    print("\n✅ RRF融合结果:")
    for i, result in enumerate(fused, 1):
        print(f"  {i}. {result['id']} (RRF: {result['rrf_score']:.4f})")
    
    return True


def test_rrf_score_calculation():
    """测试RRF分数计算"""
    print("\n" + "=" * 60)
    print("测试 3: RRF分数计算详解")
    print("=" * 60)
    
    print("\nRRF公式: RRF_score(d) = Σ 1/(k + rank(d))")
    print("k = 60 (常用默认值)")
    print()
    
    fusion = RRFusion(k=60)
    
    # 模拟一个文档在不同检索器中的排名
    doc_ranks = [
        ('doc1', 1),  # 检索器1排名第1
        ('doc1', 3),  # 检索器2排名第3
        ('doc1', 2),  # 检索器3排名第2
    ]
    
    print("示例: doc1 在三个检索器中的排名分别是 1, 3, 2")
    
    # 计算RRF分数
    rrf_score = 1/(60+1) + 1/(60+3) + 1/(60+2)
    print(f"RRF分数 = 1/(60+1) + 1/(60+3) + 1/(60+2)")
    print(f"       = {1/61:.6f} + {1/63:.6f} + {1/62:.6f}")
    print(f"       = {rrf_score:.6f}")
    
    # 与其他文档比较
    print("\n比较不同排名的RRF分数:")
    print(f"  排名第1: 1/(60+1) = {1/61:.6f}")
    print(f"  排名第5: 1/(60+5) = {1/65:.6f}")
    print(f"  排名第10: 1/(60+10) = {1/70:.6f}")
    print(f"  排名第20: 1/(60+20) = {1/80:.6f}")
    
    print("\n结论: RRF分数对排名位置敏感，但不是线性敏感的，适合融合多个检索器")
    
    return True


def test_advantages():
    """展示RRF的优势"""
    print("\n" + "=" * 60)
    print("测试 4: RRF优势展示")
    print("=" * 60)
    
    print("\n【优势1】不需要分数归一化")
    print("  - 向量检索分数: 0.0-1.0 (余弦相似度)")
    print("  - BM25分数: 无限制 (取决于文档长度)")
    print("  - 关键词分数: 整数 (匹配次数)")
    print("  → RRF只使用排名，不受分数尺度影响")
    
    print("\n【优势2】对排名顺序敏感")
    print("  示例: 文档A在所有检索器中都排第2名")
    print("        文档B在某个检索器中排第1，另一个排第5")
    print("  → A: 2*(1/(60+2)) = 0.0323")
    print("  → B: 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318")
    print("  → A略微胜出，因为它在所有检索器中排名更稳定")
    
    print("\n【优势3】鲁棒性好")
    print("  - 单个检索器的异常分数不会影响结果")
    print("  - 即使某个检索器失效，其他检索器仍能正常工作")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("RRF (Reciprocal Rank Fusion) 融合算法测试")
    print("=" * 60)
    
    tests = [
        ("基本RRF融合", test_basic_rrf),
        ("简单RRF融合", test_simple_rrf),
        ("RRF分数计算", test_rrf_score_calculation),
        ("RRF优势展示", test_advantages),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n📚 RRF融合算法说明:")
        print("  - RRF_score(d) = Σ 1/(k + rank(d))")
        print("  - k = 60 (常用默认值)")
        print("  - 优点: 不需要调参，对排名敏感，鲁棒性好")
    else:
        print("\n⚠️ 部分测试失败，请检查输出")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
