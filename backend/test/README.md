# RAG系统Benchmark测试

## 测试结构

```
backend/test/
├── __init__.py
├── conftest.py                    # pytest配置和fixtures
├── data/
│   ├── __init__.py
│   └── test_cases.py              # 测试数据集（12个测试用例）
├── test_retrieval_quality.py      # 检索质量测试
├── test_end_to_end.py             # 端到端评估测试
├── test_performance.py            # 性能测试
└── test_ragas_evaluation.py       # Ragas评估测试
```

## 测试类型

### 1. 检索质量测试 (test_retrieval_quality.py)

| 指标 | 说明 | 目标 |
|------|------|------|
| Precision@5 | 前5个结果中相关文档的比例 | ≥ 0.6 |
| Recall@5 | 召回的相关文档占全部相关文档的比例 | ≥ 0.5 |
| MRR | 首个相关文档排名的倒数 | ≥ 0.5 |
| Hit Rate@5 | 前5个是否包含相关文档 | ≥ 0.8 |
| Latency P95 | 检索延迟95百分位 | < 200ms |

### 2. 端到端评估测试 (test_end_to_end.py)

| 指标 | 说明 | 目标 |
|------|------|------|
| Keyword Coverage | 答案中关键词覆盖率 | ≥ 0.5 |
| Citation Precision | 引用的参考文献是否相关 | ≥ 0.7 |
| Citation Recall | 应该引用的文档是否都被引用 | ≥ 0.6 |
| Hallucination Rate | 答案中不在文档中的内容比例 | < 0.3 |

### 3. 性能测试 (test_performance.py)

| 指标 | 说明 | 目标 |
|------|------|------|
| Latency P50 | 典型检索延迟 | < 100ms |
| Latency P95 | 极端情况检索延迟 | < 200ms |
| Throughput (QPS) | 每秒处理查询数 | ≥ 5 |
| Concurrent Latency | 并发请求最大延迟 | < 500ms |

### 4. Ragas评估测试 (test_ragas_evaluation.py)

| 指标 | 说明 | 目标 |
|------|------|------|
| Context Precision | 检索上下文的精确度 | ≥ 0.5 |
| Context Recall | 检索上下文的召回率 | ≥ 0.5 |
| Faithfulness | 答案对上下文的忠实度 | ≥ 0.7 |
| Answer Relevance | 答案与问题的相关性 | ≥ 0.6 |

## 运行方式

```bash
cd d:\abcd\ai_coding\ai_coding_website\backend

# 运行所有测试
uv run pytest test/ -v

# 运行特定测试文件
uv run pytest test/test_retrieval_quality.py -v
uv run pytest test/test_end_to_end.py -v
uv run pytest test/test_performance.py -v
uv run pytest test/test_ragas_evaluation.py -v

# 运行特定测试类
uv run pytest test/test_retrieval_quality.py::TestRetrievalQuality -v

# 运行特定测试方法
uv run pytest test/test_retrieval_quality.py::TestRetrievalQuality::test_retrieval_quality_summary -v

# 生成详细报告
uv run pytest test/ -v --tb=short

# 只运行ragas测试
uv run pytest test/test_ragas_evaluation.py -v -m "not skip"
```

## 测试数据集

测试数据集位于 `test/data/test_cases.py`，包含12个测试用例：

```python
RAG_BENCHMARK_TEST_CASES = [
    {
        "id": 1,
        "query": "抢救危重患者时应该注意什么",
        "expected_keywords": ["抢救记录", "抢救措施", "及时记录", "病历书写"],
        "expected_sources": ["电子病历怎么填.txt"],
        "category": "医疗规范"
    },
    # ... 更多测试用例
]
```

## Ragas评估说明

Ragas是一个专门用于评估RAG系统的框架，提供以下评估指标：

1. **Context Precision** - 检索的上下文是否精确
2. **Context Recall** - 检索的上下文是否召回了所有相关信息
3. **Faithfulness** - 生成的答案是否忠实于检索的上下文
4. **Answer Relevance** - 答案与问题的相关性

### Ragas依赖

运行ragas测试需要安装以下依赖：

```bash
uv add ragas
uv add datasets
uv add langchain-openai  # 或 langchain-anthropic
```

## 注意事项

1. 运行测试前需要确保RAG文档已正确上传到Qdrant
2. 确保数据库连接正常
3. Ragas测试需要配置LLM（OpenAI或Anthropic）
4. 性能测试可能需要较长时间，建议在稳定环境中运行
