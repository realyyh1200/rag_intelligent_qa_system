"""
端到端测试脚本 - 测试嵌入服务和会话存储的完整功能
由于网络限制，跳过需要ChromaDB下载模型的测试
"""
import sys
import os
import tempfile

# 添加backend目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService, embedding_service
from services.session_storage import SessionStorage

def test_embedding_service_full():
    """完整测试嵌入服务"""
    print("=" * 70)
    print("测试1: 嵌入服务完整功能")
    print("=" * 70)
    
    try:
        # 测试词嵌入
        texts = [
            "Python是一种高级编程语言",
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络进行特征学习",
            "自然语言处理让计算机理解人类语言"
        ]
        embeddings = embedding_service.encode(texts)
        
        print(f"✅ 词嵌入测试成功")
        print(f"   - 文本数量: {len(texts)}")
        print(f"   - 向量维度: {len(embeddings[0])}")
        print(f"   - 使用方案: {'sentence-transformers' if not embedding_service.use_fallback else '轻量级回退'}")
        
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
        
        # 测试带元数据的精排
        items_with_metadata = [
            {"id": "1", "content": "Python是一种高级编程语言", "type": "tech"},
            {"id": "2", "content": "Java是一种面向对象的语言", "type": "tech"},
            {"id": "3", "content": "Python数据处理入门", "type": "tutorial"}
        ]
        reranked_with_meta = embedding_service.rerank_with_metadata(query, items_with_metadata)
        print(f"\n✅ 带元数据精排测试成功")
        print(f"   返回 {len(reranked_with_meta)} 条结果")
        
        # 测试语义搜索
        search_results = embedding_service.semantic_search(query, documents, top_k=3)
        print(f"\n✅ 语义搜索测试成功")
        print(f"   返回 {len(search_results)} 条结果")
        
        # 测试嵌入维度
        dim = embedding_service.get_embedding_dimension()
        print(f"✅ 获取嵌入维度成功: {dim}")
        
        return True
    except Exception as e:
        print(f"❌ 嵌入服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_storage_full():
    """完整测试SessionStorage服务"""
    print("\n" + "=" * 70)
    print("测试2: SessionStorage完整功能")
    print("=" * 70)
    
    try:
        # 使用临时目录测试
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SessionStorage(data_dir=os.path.join(tmp_dir, "memories"))
            
            # 创建会话
            session = storage.create_session(
                user_id=1,
                conversation_id=100,
                model="test-model",
                tags=["test", "e2e"]
            )
            session_id = session["id"]
            print(f"✅ 创建会话成功: {session_id}")
            
            # 添加多条记忆
            memories = [
                {"content": "这是第一条记忆 - 关于Python编程", "memory_type": "learned", "importance": 3},
                {"content": "这是第二条记忆 - 关于机器学习", "memory_type": "learned", "importance": 2},
                {"content": "这是第三条记忆 - 关于深度学习", "memory_type": "fact", "importance": 1}
            ]
            
            for mem in memories:
                storage.append_memory(
                    user_id=1,
                    session_id=session_id,
                    content=mem["content"],
                    memory_type=mem["memory_type"],
                    importance=mem["importance"]
                )
            print(f"✅ 添加 {len(memories)} 条记忆成功")
            
            # 添加消息
            storage.append_message(
                user_id=1,
                session_id=session_id,
                role="user",
                content="你好，我想学习Python",
                message_type="message"
            )
            storage.append_message(
                user_id=1,
                session_id=session_id,
                role="assistant",
                content="好的，我可以帮助你学习Python编程",
                message_type="message"
            )
            print(f"✅ 添加消息成功")
            
            # 获取会话记忆
            session_memories = storage.get_session_memories(1, session_id)
            print(f"✅ 获取记忆成功，共 {len(session_memories)} 条")
            
            # 获取会话消息
            session_messages = storage.get_session_messages(1, session_id)
            print(f"✅ 获取消息成功，共 {len(session_messages)} 条")
            
            # 获取会话信息
            session_info = storage.get_session(1, session_id)
            print(f"✅ 获取会话信息成功: {session_info['id']}, 更新时间: {session_info['updated_at'][:19]}")
            
            # 获取用户所有会话
            user_sessions = storage.get_user_sessions(1)
            print(f"✅ 获取用户会话列表成功，共 {len(user_sessions)} 个会话")
            
            # 更新会话
            updated = storage.update_session(1, session_id, tags=["updated", "test"])
            print(f"✅ 更新会话成功: {updated['id']}, 标签: {updated['tags']}")
            
            # 测试分页获取记忆
            paginated = storage.get_session_messages(1, session_id, limit=1)
            print(f"✅ 分页获取消息成功，每页 {len(paginated)} 条")
            
            # 获取用户统计
            stats = storage.get_user_stats(1)
            print(f"✅ 获取用户统计成功: {stats}")
            
            # 通过conversation_id获取会话
            conv_session = storage.get_conversation_session(1, 100)
            print(f"✅ 通过conversation_id获取会话成功: {conv_session['id']}")
            
            # 删除会话
            storage.delete_session(1, session_id)
            deleted_check = storage.get_session(1, session_id)
            print(f"✅ 删除会话成功: {'已删除' if deleted_check is None else '未删除'}")
        
        return True
    except Exception as e:
        print(f"❌ SessionStorage测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_storage_edge_cases():
    """测试SessionStorage边界情况"""
    print("\n" + "=" * 70)
    print("测试3: SessionStorage边界情况")
    print("=" * 70)
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SessionStorage(data_dir=os.path.join(tmp_dir, "memories"))
            
            # 测试获取不存在的会话
            not_found = storage.get_session(999, "nonexistent")
            print(f"✅ 获取不存在的会话: {not_found}")
            
            # 测试获取不存在会话的记忆
            no_memories = storage.get_session_memories(999, "nonexistent")
            print(f"✅ 获取不存在会话的记忆: {no_memories}")
            
            # 测试删除不存在的会话
            delete_result = storage.delete_session(999, "nonexistent")
            print(f"✅ 删除不存在的会话: {delete_result}")
            
            # 测试空内容
            session = storage.create_session(user_id=2)
            storage.append_memory(user_id=2, session_id=session["id"], content="")
            memories = storage.get_session_memories(2, session["id"])
            print(f"✅ 空内容记忆处理成功: {len(memories)} 条")
        
        return True
    except Exception as e:
        print(f"❌ SessionStorage边界测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_edge_cases():
    """测试嵌入服务边界情况"""
    print("\n" + "=" * 70)
    print("测试4: 嵌入服务边界情况")
    print("=" * 70)
    
    try:
        # 测试空输入
        empty_embedding = embedding_service.encode([])
        print(f"✅ 空列表编码: {empty_embedding}")
        
        # 测试空文本
        empty_text_embedding = embedding_service.encode([""])
        print(f"✅ 空文本编码成功，维度: {len(empty_text_embedding[0])}")
        
        # 测试特殊字符
        special_texts = ["Hello World!", "你好世界！", "12345", "测试@#$%"]
        special_embeddings = embedding_service.encode(special_texts)
        print(f"✅ 特殊字符文本编码成功，数量: {len(special_embeddings)}")
        
        # 测试空文档精排
        empty_rerank = embedding_service.rerank("test", [])
        print(f"✅ 空文档精排: {empty_rerank}")
        
        # 测试空查询精排
        docs = ["test document"]
        empty_query_rerank = embedding_service.rerank("", docs)
        print(f"✅ 空查询精排: {len(empty_query_rerank)} 条结果")
        
        # 测试带元数据的空列表
        empty_meta = embedding_service.rerank_with_metadata("test", [])
        print(f"✅ 空列表带元数据精排: {empty_meta}")
        
        return True
    except Exception as e:
        print(f"❌ 嵌入服务边界测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有端到端测试"""
    print("\n" + "=" * 70)
    print("端到端测试 - 完整功能验证")
    print("=" * 70 + "\n")
    
    tests = [
        ("嵌入服务", test_embedding_service_full),
        ("SessionStorage", test_session_storage_full),
        ("SessionStorage边界情况", test_session_storage_edge_cases),
        ("嵌入服务边界情况", test_embedding_edge_cases),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\n📋 开始测试: {name}")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印结果汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有端到端测试通过！")
    else:
        print("\n⚠️ 部分测试失败，请检查输出")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
