import pytest
import time
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestPerformance:
    @pytest.fixture(autouse=True)
    def setup(self, rag_service, benchmark_test_cases, setup_test_data):
        self.rag_service = rag_service
        self.test_cases = benchmark_test_cases
        self.metrics = defaultdict(list)

    def _measure_retrieval_latency(self, query, iterations=5):
        latencies = []
        for _ in range(iterations):
            start_time = time.time()
            self.rag_service.retrieve(query, top_k=5)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
        return latencies

    def test_retrieval_latency_p50(self):
        for case in self.test_cases:
            latencies = self._measure_retrieval_latency(case["query"], iterations=3)
            self.metrics["latency_p50"].append(np.percentile(latencies, 50))

        mean_latency = np.mean(self.metrics["latency_p50"])

        print(f"\n[Retrieval Latency P50] Mean: {mean_latency:.2f}ms")

        assert mean_latency < 100, f"Retrieval P50 latency too high: {mean_latency:.2f}ms"

    def test_retrieval_latency_p95(self):
        for case in self.test_cases:
            latencies = self._measure_retrieval_latency(case["query"], iterations=5)
            self.metrics["latency_p95"].append(np.percentile(latencies, 95))

        mean_latency = np.mean(self.metrics["latency_p95"])

        print(f"\n[Retrieval Latency P95] Mean: {mean_latency:.2f}ms")

        assert mean_latency < 200, f"Retrieval P95 latency too high: {mean_latency:.2f}ms"

    def test_retrieval_latency_distribution(self):
        all_latencies = []

        for case in self.test_cases:
            latencies = self._measure_retrieval_latency(case["query"], iterations=10)
            all_latencies.extend(latencies)

        p50 = np.percentile(all_latencies, 50)
        p90 = np.percentile(all_latencies, 90)
        p95 = np.percentile(all_latencies, 95)
        p99 = np.percentile(all_latencies, 99)

        print(f"\n[Retrieval Latency Distribution]")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P90: {p90:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        print(f"  Total measurements: {len(all_latencies)}")

        assert p95 < 200, f"Retrieval P95 latency too high: {p95:.2f}ms"

    def test_retrieval_throughput(self):
        queries = [case["query"] for case in self.test_cases[:5]]
        duration_seconds = 5
        request_count = 0
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            for query in queries:
                self.rag_service.retrieve(query, top_k=5)
                request_count += 1

        actual_duration = time.time() - start_time
        qps = request_count / actual_duration

        print(f"\n[Retrieval Throughput]")
        print(f"  QPS: {qps:.2f}")
        print(f"  Total requests: {request_count}")
        print(f"  Duration: {actual_duration:.2f}s")

        assert qps >= 5, f"Throughput too low: {qps:.2f} QPS"

    def test_concurrent_retrieval(self):
        queries = [case["query"] for case in self.test_cases]
        num_threads = 4
        results_per_thread = []
        latencies = []

        def retrieve_batch(query):
            start_time = time.time()
            result = self.rag_service.retrieve(query, top_k=5)
            latency = (time.time() - start_time) * 1000
            return latency, len(result)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(retrieve_batch, q) for q in queries * 2]

            for future in as_completed(futures):
                latency, result_count = future.result()
                latencies.append(latency)
                results_per_thread.append(result_count)

        mean_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        print(f"\n[Concurrent Retrieval ({num_threads} threads)]")
        print(f"  Mean latency: {mean_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        print(f"  Total requests: {len(latencies)}")

        assert max_latency < 500, f"Max concurrent latency too high: {max_latency:.2f}ms"

    def test_model_warmup_latency(self):
        query = self.test_cases[0]["query"]

        cold_latencies = []
        for _ in range(3):
            start_time = time.time()
            self.rag_service.retrieve(query, top_k=5)
            cold_latencies.append((time.time() - start_time) * 1000)

        warmup_latencies = []
        for _ in range(5):
            start_time = time.time()
            self.rag_service.retrieve(query, top_k=5)
            warmup_latencies.append((time.time() - start_time) * 1000)

        cold_mean = np.mean(cold_latencies)
        warmup_mean = np.mean(warmup_latencies)

        print(f"\n[Warmup Effect]")
        print(f"  Cold start mean: {cold_mean:.2f}ms")
        print(f"  Warm mean: {warmup_mean:.2f}ms")
        print(f"  Improvement: {(cold_mean - warmup_mean) / cold_mean * 100:.1f}%")

    def test_performance_summary(self):
        all_latencies = []

        for case in self.test_cases:
            latencies = self._measure_retrieval_latency(case["query"], iterations=10)
            all_latencies.extend(latencies)

        summary = {
            "latency": {
                "mean": np.mean(all_latencies),
                "std": np.std(all_latencies),
                "min": np.min(all_latencies),
                "max": np.max(all_latencies),
                "p50": np.percentile(all_latencies, 50),
                "p90": np.percentile(all_latencies, 90),
                "p95": np.percentile(all_latencies, 95),
                "p99": np.percentile(all_latencies, 99),
            }
        }

        print("\n" + "=" * 60)
        print("RAG PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"\nRetrieval Latency (ms):")
        for stat_name, stat_value in summary["latency"].items():
            print(f"  {stat_name}: {stat_value:.2f}")
        print(f"\nTotal test cases: {len(self.test_cases)}")
        print(f"Total latency measurements: {len(all_latencies)}")
        print("=" * 60)
