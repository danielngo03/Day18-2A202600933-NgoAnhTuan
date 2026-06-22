from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, math, re, hashlib
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
                    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokenize(chunk.get("text", "")) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=score,
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        self.client = None
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception as exc:
            print(f"  ⚠️  Qdrant unavailable, using in-memory dense fallback: {exc}")
        self._encoder = None
        self._memory_chunks: list[dict] = []
        self._memory_vectors: list[list[float]] = []

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as exc:
                if OPENAI_API_KEY:
                    print(f"  ⚠️  Local embedding model unavailable, using OpenAI embeddings: {exc}")
                    self._encoder = _OpenAIEncoder(OPENAI_EMBEDDING_MODEL, EMBEDDING_DIM)
                else:
                    print(f"  ⚠️  Embedding model unavailable, using lexical dense fallback: {exc}")
                    self._encoder = _HashingEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        texts = [c.get("text", "") for c in chunks]
        vectors = self._encode_texts(texts)
        self._memory_chunks = chunks
        self._memory_vectors = vectors

        if self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=len(vectors[0]) if vectors else EMBEDDING_DIM, distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")},
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            if points:
                self.client.upsert(collection, points=points)
        except Exception as exc:
            print(f"  ⚠️  Qdrant index failed, kept in-memory dense index: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._encode_texts([query])[0]
        if self.client is not None:
            try:
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=pt.payload.get("text", ""),
                        score=float(pt.score),
                        metadata={k: v for k, v in pt.payload.items() if k != "text"},
                        method="dense",
                    )
                    for pt in response.points
                ]
            except Exception as exc:
                print(f"  ⚠️  Qdrant search failed, using in-memory dense fallback: {exc}")
        return self._memory_search(query_vector, top_k)

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        encoded = self._get_encoder().encode(texts, show_progress_bar=False) if texts else []
        vectors = []
        for vector in encoded:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            vectors.append([float(x) for x in vector])
        return vectors

    def _memory_search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        scored = []
        for chunk, vector in zip(self._memory_chunks, self._memory_vectors):
            scored.append((_cosine(query_vector, vector), chunk))
        return [
            SearchResult(
                text=chunk.get("text", ""),
                score=float(score),
                metadata=chunk.get("metadata", {}),
                method="dense",
            )
            for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
            if score > 0
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in merged
    ]


def _tokenize(text: str) -> list[str]:
    segmented = segment_vietnamese(text).lower()
    return re.findall(r"[\wÀ-ỹ]+", segmented, flags=re.UNICODE)


class _SimpleBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / max(1, len(corpus_tokens))
        self.doc_freq: dict[str, int] = {}
        for doc in corpus_tokens:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.n_docs = len(corpus_tokens)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for doc in self.corpus_tokens:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            doc_len = len(doc)
            score = 0.0
            for token in query_tokens:
                if token not in counts:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                tf = counts[token]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1e-9, self.avgdl))
                score += idf * (tf * (self.k1 + 1) / denom)
            scores.append(score)
        return scores


class _HashingEncoder:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, show_progress_bar: bool = False):
        if isinstance(texts, str):
            return _hash_vector(texts, self.dim)
        return [_hash_vector(text, self.dim) for text in texts]


class _OpenAIEncoder:
    def __init__(self, model: str, dim: int):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.dim = dim

    def encode(self, texts, show_progress_bar: bool = False):
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        response = self.client.embeddings.create(
            model=self.model,
            input=inputs,
            dimensions=self.dim,
        )
        vectors = [item.embedding for item in response.data]
        return vectors[0] if single else vectors


def _hash_vector(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in _tokenize(text):
        vector[_stable_bucket(token, dim)] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / max(1e-9, norm_a * norm_b)


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % dim


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
