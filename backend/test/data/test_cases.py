RAG_BENCHMARK_TEST_CASES = [
    {
        "id": 1,
        "query": "抢救危重患者时应该注意什么",
        "expected_keywords": ["抢救记录", "抢救措施", "及时记录", "病历书写"],
        "expected_sources": ["test_document.txt"],
        "category": "医疗规范"
    },
    {
        "id": 2,
        "query": "门急诊病历包括哪些内容",
        "expected_keywords": ["门诊病历", "急诊病历", "健康体检", "医学检验"],
        "expected_sources": ["test_document.txt"],
        "category": "病历内容"
    },
    {
        "id": 3,
        "query": "住院病历首页如何填写",
        "expected_keywords": ["住院病历", "首页", "患者信息", "入院记录"],
        "expected_sources": ["test_document.txt"],
        "category": "病历填写"
    },
    {
        "id": 4,
        "query": "电子处方有什么要求",
        "expected_keywords": ["电子处方", "药品通用名称", "剂型", "剂量", "用法"],
        "expected_sources": ["test_document.txt"],
        "category": "医疗记录"
    },
    {
        "id": 5,
        "query": "手术同意书包含哪些要素",
        "expected_keywords": ["手术同意书", "患者知情", "医师签名", "麻醉"],
        "expected_sources": ["test_document.txt"],
        "category": "知情同意"
    },
    {
        "id": 6,
        "query": "医学影像检查资料如何管理",
        "expected_keywords": ["医学影像", "检查资料", "影像", "检查"],
        "expected_sources": ["test_document.txt"],
        "category": "检查管理"
    },
    {
        "id": 7,
        "query": "输血治疗知情同意书有什么规定",
        "expected_keywords": ["输血", "知情同意", "输血治疗", "输血风险"],
        "expected_sources": ["test_document.txt"],
        "category": "知情同意"
    },
    {
        "id": 8,
        "query": "护理记录应该如何书写",
        "expected_keywords": ["护理记录", "护理措施", "护理时间", "护士签名"],
        "expected_sources": ["test_document.txt"],
        "category": "医疗记录"
    },
    {
        "id": 9,
        "query": "麻醉同意书的内容要求",
        "expected_keywords": ["麻醉同意书", "麻醉方式", "麻醉风险", "患者签名"],
        "expected_sources": ["test_document.txt"],
        "category": "知情同意"
    },
    {
        "id": 10,
        "query": "辅助检查报告单包括什么",
        "expected_keywords": ["辅助检查", "检查报告", "检查结果", "报告单"],
        "expected_sources": ["test_document.txt"],
        "category": "检查管理"
    },
    {
        "id": 11,
        "query": "患者信息包括哪些内容",
        "expected_keywords": ["患者姓名", "性别", "年龄", "身份证号"],
        "expected_sources": ["test_document.txt"],
        "category": "医疗规范"
    },
    {
        "id": 12,
        "query": "病历保存期限是多久",
        "expected_keywords": ["病历保存", "保存期限", "门诊病历", "住院病历"],
        "expected_sources": ["test_document.txt"],
        "category": "病历管理"
    }
]

EXPECTED_METRICS = {
    "retrieval": {
        "precision_at_5": 0.8,
        "recall_at_5": 0.7,
        "hit_rate_at_5": 0.9,
        "mrr": 0.75
    },
    "generation": {
        "answer_accuracy": 0.8,
        "citation_precision": 0.85,
        "citation_recall": 0.8,
        "hallucination_rate": 0.1
    },
    "performance": {
        "retrieval_latency_p50_ms": 50,
        "retrieval_latency_p95_ms": 100,
        "end_to_end_latency_p50_ms": 3000,
        "end_to_end_latency_p95_ms": 8000
    }
}
