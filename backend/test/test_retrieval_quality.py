import pytest
import time
from collections import defaultdict
import numpy as np


class TestRetrievalQuality:
    @pytest.fixture(autouse=True)
    def setup(self, rag_service, benchmark_test_cases, setup_test_data):
        self.rag_service = rag_service
        self.test_cases = benchmark_test_cases
        self.metrics = defaultdict(list)

    def _calculate_precision_at_k(self, results, expected_sources, k):
        relevant = sum(1 for r in results[:k] if r.get("file_name") in expected_sources)
        return relevant / k if k > 0 else 0

    def _calculate_recall_at_k(self, results, expected_sources, k, total_relevant):
        relevant = sum(1 for r in results[:k] if r.get("file_name") in expected_sources)
        return relevant / total_relevant if total_relevant > 0 else 0

    def _calculate_mrr(self, results, expected_sources):
        for i, r in enumerate(results, 1):
            if r.get("file_name") in expected_sources:
                return 1.0 / i
        return 0.0

    def _calculate_hit_rate(self, results, expected_sources):
        return 1.0 if any(r.get("file_name") in expected_sources for r in results) else 0.0

    def _retrieve_and_evaluate(self, query, expected_sources):
        start_time = time.time()
        results = self.rag_service.retrieve(query, top_k=5)
        retrieval_time = (time.time() - start_time) * 1000

        precision = self._calculate_precision_at_k(results, expected_sources, 5)
        recall = self._calculate_recall_at_k(results, expected_sources, 5, len(expected_sources))
        mrr = self._calculate_mrr(results, expected_sources)
        hit_rate = self._calculate_hit_rate(results, expected_sources)

        return {
            "precision@5": precision,
            "recall@5": recall,
            "mrr": mrr,
            "hit_rate@5": hit_rate,
            "latency_ms": retrieval_time,
            "results_count": len(results)
        }

    def test_retrieval_precision_at_5(self):
        for case in self.test_cases:
            metrics = self._retrieve_and_evaluate(case["query"], case["expected_sources"])
            self.metrics["precision@5"].append(metrics["precision@5"])

        precision_mean = np.mean(self.metrics["precision@5"])
        precision_std = np.std(self.metrics["precision@5"])

        print(f"\n[Precision@5] Mean: {precision_mean:.4f}, Std: {precision_std:.4f}")
        print(f"  Min: {min(self.metrics['precision@5']):.4f}, Max: {max(self.metrics['precision@5']):.4f}")

        assert precision_mean >= 0.55, f"Precision@5 too low: {precision_mean:.4f}"

    def test_retrieval_recall_at_5(self):
        for case in self.test_cases:
            metrics = self._retrieve_and_evaluate(case["query"], case["expected_sources"])
            self.metrics["recall@5"].append(metrics["recall@5"])

        recall_mean = np.mean(self.metrics["recall@5"])
        recall_std = np.std(self.metrics["recall@5"])

        print(f"\n[Recall@5] Mean: {recall_mean:.4f}, Std: {recall_std:.4f}")
        print(f"  Min: {min(self.metrics['recall@5']):.4f}, Max: {max(self.metrics['recall@5']):.4f}")

        assert recall_mean >= 0.5, f"Recall@5 too low: {recall_mean:.4f}"

    def test_retrieval_mrr(self):
        for case in self.test_cases:
            metrics = self._retrieve_and_evaluate(case["query"], case["expected_sources"])
            self.metrics["mrr"].append(metrics["mrr"])

        mrr_mean = np.mean(self.metrics["mrr"])
        mrr_std = np.std(self.metrics["mrr"])

        print(f"\n[MRR] Mean: {mrr_mean:.4f}, Std: {mrr_std:.4f}")
        print(f"  Min: {min(self.metrics['mrr']):.4f}, Max: {max(self.metrics['mrr']):.4f}")

        assert mrr_mean >= 0.5, f"MRR too low: {mrr_mean:.4f}"

    def test_retrieval_hit_rate_at_5(self):
        for case in self.test_cases:
            metrics = self._retrieve_and_evaluate(case["query"], case["expected_sources"])
            self.metrics["hit_rate@5"].append(metrics["hit_rate@5"])

        hit_rate_mean = np.mean(self.metrics["hit_rate@5"])

        print(f"\n[Hit Rate@5] Mean: {hit_rate_mean:.4f}")
        print(f"  Hits: {sum(self.metrics['hit_rate@5'])}/{len(self.metrics['hit_rate@5'])}")

        assert hit_rate_mean >= 0.8, f"Hit Rate@5 too low: {hit_rate_mean:.4f}"

    def test_retrieval_latency(self):
        for case in self.test_cases:
            metrics = self._retrieve_and_evaluate(case["query"], case["expected_sources"])
            self.metrics["latency_ms"].append(metrics["latency_ms"])

        latency_p50 = np.percentile(self.metrics["latency_ms"], 50)
        latency_p95 = np.percentile(self.metrics["latency_ms"], 95)
        latency_mean = np.mean(self.metrics["latency_ms"])

        print(f"\n[Retrieval Latency]")
        print(f"  Mean: {latency_mean:.2f}ms, P50: {latency_p50:.2f}ms, P95: {latency_p95:.2f}ms")

        assert latency_p95 < 3000, f"Retrieval P95 latency too high: {latency_p95:.2f}ms"

    def test_all_cases_have_results(self):
        for case in self.test_cases:
            metrics = self._retrieve_and_evaluate(case["query"], case["expected_sources"])
            self.metrics["results_count"].append(metrics["results_count"])

        zero_results = [i for i, c in enumerate(self.metrics["results_count"]) if c == 0]

        print(f"\n[Results Count]")
        print(f"  Mean: {np.mean(self.metrics['results_count']):.1f}")
        print(f"  Zero results cases: {len(zero_results)}")

        assert len(zero_results) == 0, f"Found {len(zero_results)} queries with zero results"

    def test_retrieval_quality_summary(self):
        for case in self.test_cases:
            metrics = self._retrieve_and_evaluate(case["query"], case["expected_sources"])
            for key, value in metrics.items():
                self.metrics[key].append(value)

        summary = {}
        for metric, values in self.metrics.items():
            if metric == "results_count":
                continue
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95)
            }

        print("\n" + "=" * 60)
        print("RAG RETRIEVAL QUALITY SUMMARY")
        print("=" * 60)
        for metric, stats in summary.items():
            print(f"\n{metric}:")
            for stat_name, stat_value in stats.items():
                if "ms" in metric:
                    print(f"  {stat_name}: {stat_value:.2f}")
                else:
                    print(f"  {stat_name}: {stat_value:.4f}")
        print("\n" + "=" * 60)
