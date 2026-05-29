import pytest
from collections import defaultdict

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevance,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="ragas not installed")
class TestRagasEvaluation:
    @pytest.fixture(autouse=True)
    def setup(self, rag_service, benchmark_test_cases, setup_test_data):
        self.rag_service = rag_service
        self.test_cases = benchmark_test_cases
        self.metrics = defaultdict(list)

    def _prepare_ragas_dataset(self):
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for case in self.test_cases:
            results = self.rag_service.retrieve(case["query"], top_k=5)

            context_list = []
            for r in results:
                content = r.content if hasattr(r, 'content') else str(r)
                context_list.append(content)

            mock_answer = "根据检索到的文档，" + " ".join(context_list[:2])[:200] if context_list else "未找到相关内容"

            ground_truth = " ".join(case["expected_keywords"])

            questions.append(case["query"])
            answers.append(mock_answer)
            contexts.append(context_list)
            ground_truths.append(ground_truth)

        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

    def test_ragas_context_precision(self):
        dataset = self._prepare_ragas_dataset()

        result = evaluate(
            dataset,
            metrics=[context_precision],
        )

        context_precision_score = result['context_precision']
        self.metrics["context_precision"].append(context_precision_score)

        print(f"\n[Ragas Context Precision] Score: {context_precision_score:.4f}")

        assert context_precision_score >= 0.5, f"Context precision too low: {context_precision_score:.4f}"

    def test_ragas_context_recall(self):
        dataset = self._prepare_ragas_dataset()

        result = evaluate(
            dataset,
            metrics=[context_recall],
        )

        context_recall_score = result['context_recall']
        self.metrics["context_recall"].append(context_recall_score)

        print(f"\n[Ragas Context Recall] Score: {context_recall_score:.4f}")

        assert context_recall_score >= 0.5, f"Context recall too low: {context_recall_score:.4f}"

    def test_ragas_faithfulness(self):
        dataset = self._prepare_ragas_dataset()

        result = evaluate(
            dataset,
            metrics=[faithfulness],
        )

        faithfulness_score = result['faithfulness']
        self.metrics["faithfulness"].append(faithfulness_score)

        print(f"\n[Ragas Faithfulness] Score: {faithfulness_score:.4f}")

        assert faithfulness_score >= 0.7, f"Faithfulness too low: {faithfulness_score:.4f}"

    def test_ragas_answer_relevance(self):
        dataset = self._prepare_ragas_dataset()

        result = evaluate(
            dataset,
            metrics=[answer_relevance],
        )

        answer_relevance_score = result['answer_relevance']
        self.metrics["answer_relevance"].append(answer_relevance_score)

        print(f"\n[Ragas Answer Relevance] Score: {answer_relevance_score:.4f}")

        assert answer_relevance_score >= 0.6, f"Answer relevance too low: {answer_relevance_score:.4f}"

    def test_ragas_full_evaluation(self):
        dataset = self._prepare_ragas_dataset()

        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevance,
            ],
        )

        print("\n" + "=" * 60)
        print("RAGAS FULL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nContext Precision:  {result['context_precision']:.4f}")
        print(f"Context Recall:     {result['context_recall']:.4f}")
        print(f"Faithfulness:       {result['faithfulness']:.4f}")
        print(f"Answer Relevance:   {result['answer_relevance']:.4f}")
        print("\n" + "=" * 60)

        assert result['context_precision'] >= 0.5, "Context precision too low"
        assert result['context_recall'] >= 0.5, "Context recall too low"
        assert result['faithfulness'] >= 0.7, "Faithfulness too low"
        assert result['answer_relevance'] >= 0.6, "Answer relevance too low"


class TestRagasMetricsExplanation:
    def test_ragas_metrics_overview(self):
        print("\n" + "=" * 60)
        print("RAGAS METRICS EXPLANATION")
        print("=" * 60)
        print("""
1. Context Precision (上下文精确度)
   - 衡量：检索到的上下文中有多少与问题相关
   - 计算：相关上下文数 / 总检索上下文数
   - 目标：越高越好（接近1.0）

2. Context Recall (上下文召回率)
   - 衡量：所有相关信息是否都被检索到
   - 计算：检索到的相关信息 / 总相关信息
   - 目标：越高越好（接近1.0）

3. Faithfulness (忠实度)
   - 衡量：生成的答案是否忠实于检索的上下文
   - 计算：答案中的陈述能否被上下文支持
   - 目标：越高越好（接近1.0），避免幻觉

4. Answer Relevance (答案相关性)
   - 衡量：生成的答案与问题的相关程度
   - 计算：答案是否回答了问题的核心
   - 目标：越高越好（接近1.0）
        """)
        print("=" * 60)
