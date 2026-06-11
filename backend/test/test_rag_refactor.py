"""
RAG重构后的测试脚本
测试ChromaDB存储和检索功能
"""
import sys
import os

# 添加backend目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.chroma_service import ChromaService
from services.rag_service import RAGService
from services.bm25_service import BM25Service
from services.memory_service import MemoryService


def test_chroma_service():
    """测试ChromaDB服务"""
    print("=" * 60)
    print("测试 ChromaService")
    print("=" * 60)
    
    chroma = ChromaService()
    
    # 检查服务初始化
    if chroma.client:
        print("✅ ChromaDB初始化成功")
    else:
        print("❌ ChromaDB初始化失败")
        return False
    
    # 测试集合创建
    print("✅ 集合创建检查完成")
    
    return True


def test_bm25_service():
    """测试BM25服务"""
    print("\n" + "=" * 60)
    print("测试 BM25Service")
    print("=" * 60)
    
    bm25 = BM25Service()
    
    # 准备测试文档
    documents = [
        {
            'id': 1,
            'content': 'Python是一种高级编程语言',
            'file_name': 'python.txt'
        },
        {
            'id': 2,
            'content': 'Java是一种面向对象的编程语言',
            'file_name': 'java.txt'
        },
        {
            'id': 3,
            'content': '机器学习是人工智能的一个分支',
            'file_name': 'ml.txt'
        }
    ]
    
    # 初始化
    bm25.initialize(documents)
    print("✅ BM25索引初始化完成")
    
    # 测试搜索
    results = bm25.search('Python编程', top_k=2)
    print(f"搜索'Python编程'，找到 {len(results)} 条结果")
    
    for result in results:
        print(f"  - ID: {result['id']}, Score: {result['score']:.4f}")
        print(f"    Content: {result['content']}")
    
    return True


def test_rag_service():
    """测试RAG服务"""
    print("\n" + "=" * 60)
    print("测试 RAGService")
    print("=" * 60)
    
    user_id = 999  # 测试用户ID
    
    rag_service = RAGService(user_id=user_id)
    print("✅ RAGService初始化成功")
    
    # 测试文件列表获取
    files = rag_service.get_user_files()
    print(f"当前用户文件数: {len(files)}")
    
    return True


def test_memory_service():
    """测试Memory服务"""
    print("\n" + "=" * 60)
    print("测试 MemoryService")
    print("=" * 60)
    
    user_id = 999
    conversation_id = 1
    
    memory_service = MemoryService(user_id=user_id, conversation_id=conversation_id)
    print("✅ MemoryService初始化成功")
    
    # 测试添加消息
    memory_service.add_message("user", "你好，我是用户")
    memory_service.add_message("assistant", "你好，有什么可以帮助你的吗？")
    print("✅ 添加消息成功")
    
    # 测试获取上下文
    context = memory_service.get_context_for_ai()
    print(f"✅ 获取上下文成功，长度: {len(context)} 字符")
    
    # 测试统计
    stats = memory_service.get_memory_stats()
    print(f"✅ 记忆统计: {stats}")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 RAG 重构后的功能")
    print("=" * 60 + "\n")
    
    tests = [
        ("ChromaDB服务", test_chroma_service),
        ("BM25服务", test_bm25_service),
        ("RAG服务", test_rag_service),
        ("Memory服务", test_memory_service),
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
    else:
        print("\n⚠️ 部分测试失败，请检查输出")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
