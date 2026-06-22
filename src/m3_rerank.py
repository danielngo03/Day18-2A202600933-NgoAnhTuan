from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, math, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  ⚠️  CrossEncoder unavailable, using lexical rerank fallback: {exc}")
                self._model = _LexicalCrossEncoder()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        scored = []
        for score, doc in zip(scores, documents):
            lifecycle_adjustment = 0.15 * float(doc.get("metadata", {}).get("policy_priority", 0.0))
            scored.append((float(score) + lifecycle_adjustment, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [{"id": str(i), "text": d.get("text", ""), "meta": d} for i, d in enumerate(documents)]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            ordered = []
            for rank, item in enumerate(results[:top_k]):
                doc = documents[int(item.get("id", rank))]
                ordered.append(RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=rank,
                ))
            return ordered
        except Exception:
            return CrossEncoderReranker(model_name="lexical").rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


class _LexicalCrossEncoder:
    def predict(self, pairs):
        return [_lexical_score(query, document) for query, document in pairs]


def _lexical_score(query: str, document: str) -> float:
    query_terms = _weighted_terms(query)
    doc_terms = _weighted_terms(document)
    if not query_terms or not doc_terms:
        return 0.0
    dot = sum(weight * doc_terms.get(term, 0.0) for term, weight in query_terms.items())
    q_norm = math.sqrt(sum(value * value for value in query_terms.values()))
    d_norm = math.sqrt(sum(value * value for value in doc_terms.values()))
    overlap_bonus = len(set(query_terms) & set(doc_terms)) / max(1, len(query_terms))
    return dot / max(1e-9, q_norm * d_norm) + 0.15 * overlap_bonus


def _weighted_terms(text: str) -> dict[str, float]:
    tokens = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return {token: 1.0 + math.log(count) for token, count in counts.items()}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
