"""
测试脚本 - 验证 sentence-transformers 嵌入服务和精排功能
"""
import sys
import os

# 添加backend目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService, embedding_service
from services.session_storage import SessionStorage

def test_embedding_service():
    """测试嵌入服务"""
    print("=" * 60)
    print("测试嵌入服务 (sentence-transformers)")
    print("=" * 60)
    
    try:
        # 测试词嵌入
        texts = ["Python是一种高级编程语言", "机器学习是人工智能的一个分支"]
        embeddings = embedding_service.encode(texts)
        
        print(f"✅ 词嵌入成功")
        print(f"   - 文本数量: {len(texts)}")
        print(f"   - 向量维度: {len(embeddings[0])}")
        print(f"   - 第一个向量前5个值: {embeddings[0][:5]}")
        
        # 测试单文本编码
        single_embedding = embedding_service.encode_single("测试单文本编码")
        print(f"✅ 单文本编码成功，维度: {len(single_embedding)}")
        
        # 测试精排
        query = "Python编程"
        documents = [
            "Python是一种高级编程语言，简洁易读",
            "Java是一种面向对象的编程语言",
            "机器学习使用Python进行数据处理",
            "JavaScript用于Web开发",
            "深度学习框架如PyTorch使用Python"
        ]
        
        reranked = embedding_service.rerank(query, documents)
        print(f"\n✅ 精排测试成功")
        print(f"   查询: {query}")
        for i, item in enumerate(reranked[:3], 1):
            print(f"   {i}. {item['content'][:50]}... (分数: {item['relevance_score']:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ 嵌入服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_storage():
    """测试 SessionStorage 服务"""
    print("\n" + "=" * 60)
    print("测试 SessionStorage 服务")
    print("=" * 60)
    
    try:
        storage = SessionStorage()
        
        # 创建会话
        session = storage.create_session(
            user_id=1,
            conversation_id=100,
            model="test",
            tags=["test"]
        )
        session_id = session["id"]
        print(f"✅ 创建会话成功: {session_id}")
        
        # 添加记忆
        storage.append_memory(
            user_id=1,
            session_id=session_id,
            content="测试会话记忆内容 - 使用 sentence-transformers",
            memory_type="learned",
            importance=3
        )
        print("✅ 添加记忆成功")
        
        # 获取会话记忆
        memories = storage.get_session_memories(1, session_id)
        print(f"✅ 获取记忆成功，共 {len(memories)} 条")
        
        # 获取会话信息
        session_info = storage.get_session(1, session_id)
        print(f"✅ 获取会话信息成功: {session_info['id']}")
        
        return True
    except Exception as e:
        print(f"❌ SessionStorage测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 sentence-transformers 嵌入服务")
    print("=" * 60 + "\n")
    
    tests = [
        ("嵌入服务", test_embedding_service),
        ("SessionStorage服务", test_session_storage),
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
