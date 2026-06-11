"""
简单测试脚本 - 验证 ChromaDB 和 SessionStorage 功能
"""
import sys
import os

# 添加backend目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import os

# 直接导入模块，避免通过 services/__init__.py
chroma_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'services', 'chroma_service.py')
spec = importlib.util.spec_from_file_location("chroma_service", chroma_path)
chroma_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chroma_module)
ChromaService = chroma_module.ChromaService

session_storage_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'services', 'session_storage.py')
spec = importlib.util.spec_from_file_location("session_storage", session_storage_path)
session_storage_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session_storage_module)
SessionStorage = session_storage_module.SessionStorage

def test_chroma():
    """测试 ChromaDB 服务"""
    print("=" * 60)
    print("测试 ChromaDB 服务")
    print("=" * 60)
    
    try:
        chroma = ChromaService()
        
        # 测试存储和检索
        chroma.upsert_memory(
            user_id=1,
            memory_id=1001,
            content="测试记忆内容：Python是一种高级编程语言",
            memory_type="general",
            importance=5
        )
        
        results = chroma.search_memories(
            query="Python",
            user_id=1,
            top_k=3
        )
        
        print(f"✅ 存储和检索成功，找到 {len(results)} 条结果")
        for r in results:
            print(f"   - {r['content'][:50]}... (分数: {r['relevance_score']:.3f})")
        
        # 测试RAG文档
        chroma.upsert_rag_document(
            user_id=1,
            doc_id="test_doc_1",
            content="测试文档内容：机器学习是人工智能的一个分支",
            file_name="test.txt"
        )
        
        rag_results = chroma.search_rag_documents(
            query="机器学习",
            user_id=1,
            top_k=3
        )
        
        print(f"✅ RAG文档存储和检索成功，找到 {len(rag_results)} 条结果")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB测试失败: {e}")
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
            content="测试会话记忆内容",
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
    print("开始测试新架构功能")
    print("=" * 60 + "\n")
    
    tests = [
        ("ChromaDB服务", test_chroma),
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
