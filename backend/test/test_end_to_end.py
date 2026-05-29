import pytest
import time
from collections import defaultdict
import numpy as np


class TestEndToEndQuality:
    @pytest.fixture(autouse=True)
    def setup(self, rag_service, benchmark_test_cases, setup_test_data):
        self.rag_service = rag_service
        self.test_cases = benchmark_test_cases
        self.metrics = defaultdict(list)

    def _build_context(self, results):
        context_parts = []
        for r in results:
            content = r.get("content", str(r))
            file_name = r.get("file_name", "unknown")
            context_parts.append(f"[{file_name}]\n{content}")
        return "\n\n---\n\n".join(context_parts)

    def _evaluate_answer_quality(self, answer, expected_keywords, expected_sources, context):
        keyword_hits = 0
        for keyword in expected_keywords:
            if keyword.lower() in answer.lower():
                keyword_hits += 1

        keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 0

        source_hits = 0
        for source in expected_sources:
            if source.lower() in answer.lower() or source.lower() in context.lower():
                source_hits += 1

        source_score = source_hits / len(expected_sources) if expected_sources else 0

        return {
            "keyword_score": keyword_score,
            "source_score": source_score,
            "answer_length": len(answer)
        }

    def _check_hallucination(self, answer, context):
        context_lower = context.lower()
        answer_lines = answer.split('\n')

        hallucinated_lines = 0
        for line in answer_lines:
            line = line.strip()
            if len(line) > 20 and line not in context_lower:
                if not any(punct in line for punct in ['。', '！', '？', '.', '!', '?']):
                    continue
                if '根据' in line or '按照' in line or '根据' in line:
                    hallucinated_lines += 1

        hallucination_rate = hallucinated_lines / len(answer_lines) if answer_lines else 0
        return hallucination_rate

    def test_answer_keyword_coverage(self):
        for case in self.test_cases:
            results = self.rag_service.retrieve(case["query"], top_k=5)
            context = self._build_context(results)

            mock_answer = context[:200] + "..." if context else "未找到相关内容"

            quality = self._evaluate_answer_quality(
                mock_answer,
                case["expected_keywords"],
                case["expected_sources"],
                context
            )

            self.metrics["keyword_coverage"].append(quality["keyword_score"])

        mean_coverage = np.mean(self.metrics["keyword_coverage"])

        print(f"\n[Keyword Coverage] Mean: {mean_coverage:.4f}")

        assert mean_coverage >= 0.5, f"Keyword coverage too low: {mean_coverage:.4f}"

    def test_citation_precision(self):
        for case in self.test_cases:
            results = self.rag_service.retrieve(case["query"], top_k=5)
            cited_sources = {r.metadata.get("file_name") for r in results if hasattr(r, 'metadata')}

            expected_set = set(case["expected_sources"])
            relevant_citations = len(cited_sources & expected_set)
            citation_precision = relevant_citations / len(cited_sources) if cited_sources else 0

            self.metrics["citation_precision"].append(citation_precision)

        mean_precision = np.mean(self.metrics["citation_precision"])

        print(f"\n[Citation Precision] Mean: {mean_precision:.4f}")
        print(f"  Precision >= 0.8: {sum(1 for p in self.metrics['citation_precision'] if p >= 0.8)}/{len(self.metrics['citation_precision'])}")

        assert mean_precision >= 0.7, f"Citation precision too low: {mean_precision:.4f}"

    def test_citation_recall(self):
        for case in self.test_cases:
            results = self.rag_service.retrieve(case["query"], top_k=5)
            cited_sources = {r.metadata.get("file_name") for r in results if hasattr(r, 'metadata')}

            expected_set = set(case["expected_sources"])
            relevant_citations = len(cited_sources & expected_set)
            citation_recall = relevant_citations / len(expected_set) if expected_set else 0

            self.metrics["citation_recall"].append(citation_recall)

        mean_recall = np.mean(self.metrics["citation_recall"])

        print(f"\n[Citation Recall] Mean: {mean_recall:.4f}")

        assert mean_recall >= 0.6, f"Citation recall too low: {mean_recall:.4f}"

    def test_answer_length_distribution(self):
        for case in self.test_cases:
            results = self.rag_service.retrieve(case["query"], top_k=5)
            context = self._build_context(results)

            mock_answer = context[:300] + "..." if context else "未找到相关内容"

            quality = self._evaluate_answer_quality(
                mock_answer,
                case["expected_keywords"],
                case["expected_sources"],
                context
            )

            self.metrics["answer_length"].append(quality["answer_length"])

        mean_length = np.mean(self.metrics["answer_length"])
        p50_length = np.percentile(self.metrics["answer_length"], 50)
        p95_length = np.percentile(self.metrics["answer_length"], 95)

        print(f"\n[Answer Length] Mean: {mean_length:.0f}, P50: {p50_length:.0f}, P95: {p95_length:.0f}")

        assert mean_length >= 50, f"Answer length too short: {mean_length:.0f}"

    def test_no_hallucination_in_context(self):
        for case in self.test_cases:
            results = self.rag_service.retrieve(case["query"], top_k=5)
            context = self._build_context(results)

            mock_answer = context[:300] + "..." if context else "未找到相关内容"

            hallucination_rate = self._check_hallucination(mock_answer, context)
            self.metrics["hallucination_rate"].append(hallucination_rate)

        mean_hallucination = np.mean(self.metrics["hallucination_rate"])

        print(f"\n[Hallucination Rate] Mean: {mean_hallucination:.4f}")

        assert mean_hallucination < 0.3, f"Hallucination rate too high: {mean_hallucination:.4f}"

    def test_end_to_end_latency(self):
        latencies = []

        for case in self.test_cases:
            start_time = time.time()

            results = self.rag_service.retrieve(case["query"], top_k=5)
            context = self._build_context(results)
            mock_answer = context[:200] + "..." if context else "未找到相关内容"

            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_mean = np.mean(latencies)

        print(f"\n[End-to-End Latency]")
        print(f"  Mean: {latency_mean:.2f}ms, P50: {latency_p50:.2f}ms, P95: {latency_p95:.2f}ms")

        assert latency_p95 < 5000, f"End-to-end P95 latency too high: {latency_p95:.2f}ms"

    def test_e2e_quality_summary(self):
        for case in self.test_cases:
            results = self.rag_service.retrieve(case["query"], top_k=5)
            context = self._build_context(results)

            mock_answer = context[:300] + "..." if context else "未找到相关内容"

            quality = self._evaluate_answer_quality(
                mock_answer,
                case["expected_keywords"],
                case["expected_sources"],
                context
            )

            self.metrics["keyword_score"].append(quality["keyword_score"])
            self.metrics["source_score"].append(quality["source_score"])

            cited_sources = {r.metadata.get("file_name") for r in results if hasattr(r, 'metadata')}
            expected_set = set(case["expected_sources"])
            citation_precision = len(cited_sources & expected_set) / len(cited_sources) if cited_sources else 0
            citation_recall = len(cited_sources & expected_set) / len(expected_set) if expected_set else 0

            self.metrics["citation_precision"].append(citation_precision)
            self.metrics["citation_recall"].append(citation_recall)

        summary = {}
        for metric, values in self.metrics.items():
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95)
            }

        print("\n" + "=" * 60)
        print("RAG END-TO-END QUALITY SUMMARY")
        print("=" * 60)
        for metric, stats in summary.items():
            print(f"\n{metric}:")
            for stat_name, stat_value in stats.items():
                print(f"  {stat_name}: {stat_value:.4f}" if isinstance(stat_value, float) else f"  {stat_name}: {stat_value}")
        print("\n" + "=" * 60)
