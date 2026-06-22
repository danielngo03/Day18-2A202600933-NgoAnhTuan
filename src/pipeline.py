from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import json
import os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

_QUERY_LATENCIES: list[dict] = []


def _document_version_metadata(source: str, text: str) -> dict:
    """Infer generic policy lifecycle metadata from file/content version markers."""
    source_lower = source.lower()
    text_lower = text[:800].lower()
    superseded_markers = ("superseded", "đã bị thay thế", "chính sách cũ", "hết hiệu lực")
    current_markers = ("hiện hành", "phiên bản mới", "có hiệu lực")

    is_superseded = (
        any(marker in text_lower for marker in superseded_markers)
        or source_lower.endswith("_v1.md")
        or "v2023" in source_lower
    )
    is_current = (
        any(marker in text_lower for marker in current_markers)
        or source_lower.endswith("_v2.md")
        or "v2024" in source_lower
    )
    status = "superseded" if is_superseded and not is_current else "current" if is_current else "unspecified"
    return {
        "document_status": status,
        "policy_priority": -1.0 if status == "superseded" else 1.0 if status == "current" else 0.0,
    }


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)
    _QUERY_LATENCIES.clear()

    # Step 1: Load & Chunk (M1)
    timings = {}
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        lifecycle_meta = _document_version_metadata(
            doc["metadata"].get("source", ""),
            doc["text"],
        )
        base_meta = {**doc["metadata"], **lifecycle_meta}
        parents, children = chunk_hierarchical(doc["text"], metadata=base_meta)
        parent_by_id = {p.metadata["parent_id"]: p.text for p in parents}
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id,
                    "parent_text": parent_by_id.get(child.parent_id, child.text),
                },
            })
    timings["chunking_seconds"] = round(time.time() - t0, 3)
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({timings['chunking_seconds']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        timings["enrichment_seconds"] = round(time.time() - t0, 3)
        print(f"  ✓ Enriched {len(enriched)} chunks ({timings['enrichment_seconds']:.1f}s)", flush=True)
    else:
        timings["enrichment_seconds"] = round(time.time() - t0, 3)
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    timings["indexing_seconds"] = round(time.time() - t0, 3)
    print(f"  ✓ Indexed ({timings['indexing_seconds']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    timings["reranker_load_seconds"] = round(time.time() - t0, 3)
    print(f"  ✓ Reranker ready ({timings['reranker_load_seconds']:.1f}s)", flush=True)

    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    query_start = time.perf_counter()
    step_start = time.perf_counter()
    result_by_text = {}
    for query_variant in _query_variants(query):
        for result in search.search(query_variant):
            existing = result_by_text.get(result.text)
            if existing is None or result.score > existing.score:
                result_by_text[result.text] = result
    results = list(result_by_text.values())
    search_ms = (time.perf_counter() - step_start) * 1000
    docs = [
        {
            # Score the parent because it contains the complete policy section;
            # child text remains the high-precision retrieval unit.
            "text": r.metadata.get("parent_text", r.text),
            "score": r.score,
            "metadata": r.metadata,
        }
        for r in results
    ]
    step_start = time.perf_counter()
    reranked = reranker.rerank(
        query,
        docs,
        # Rerank every fused candidate, then deduplicate parents below. A
        # single long document may contribute many child hits and must not
        # crowd out the second source required by a multi-hop question.
        top_k=len(docs),
    )
    rerank_ms = (time.perf_counter() - step_start) * 1000
    selected = reranked if reranked else results[:3]
    contexts = []
    seen = set()
    for result in selected:
        context = result.metadata.get("parent_text", result.text)
        if context not in seen:
            contexts.append(context)
            seen.add(context)
        if len(contexts) >= RERANK_TOP_K:
            break

    from config import OPENAI_API_KEY
    step_start = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {
                    "role": "system",
                    "content": (
                        "Trả lời bằng 1-3 câu đầy đủ, trực tiếp và CHỈ dựa trên context. "
                        "Câu trả lời phải tự đứng độc lập: nhắc lại chủ thể hoặc nội dung chính "
                        "của câu hỏi, trả lời đủ mọi ý được hỏi và giữ các điều kiện/ngoại lệ "
                        "quan trọng có trong context. "
                        "Nếu có nhiều phiên bản chính sách, ưu tiên bản hiện hành/mới nhất "
                        "và nêu rõ bản cũ đã bị thay thế. Với câu hỏi tính toán, trình bày phép tính. "
                        "Nếu context không đủ, nói 'Không tìm thấy.'"
                    ),
                },
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ], temperature=0)
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    generation_ms = (time.perf_counter() - step_start) * 1000
    _QUERY_LATENCIES.append({
        "search_ms": search_ms,
        "rerank_ms": rerank_ms,
        "generation_ms": generation_ms,
        "total_ms": (time.perf_counter() - query_start) * 1000,
    })
    return answer, contexts


def _query_variants(query: str) -> list[str]:
    """Create focused variants for multi-part questions without domain rules."""
    parts = [
        part.strip(" ,;?.")
        for part in re.split(r"\s+(?:và|đồng thời|ngoài ra)\s+", query, flags=re.IGNORECASE)
        if len(part.strip(" ,;?.")) >= 8
    ]
    if len(parts) < 2:
        return [query]

    words = re.findall(r"[\wÀ-ỹ-]+", query, flags=re.UNICODE)
    anchors = []
    for word in words[1:]:
        if any(char.isdigit() for char in word) or word.isupper() or (
            word[:1].isupper() and len(word) > 2
        ):
            if word.lower() not in {"VNĐ".lower()} and word not in anchors:
                anchors.append(word)

    variants = [query]
    for part in parts:
        missing_anchors = [
            anchor for anchor in anchors if anchor.lower() not in part.lower()
        ]
        variant = " ".join([part, *missing_anchors]).strip()
        if variant and variant not in variants:
            variants.append(variant)
    return variants


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(
        questions, answers, all_contexts, ground_truths, use_live=True
    )
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.70 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, path="reports/ragas_report.json")
    _save_query_latency()
    return results


def _save_query_latency() -> None:
    if not _QUERY_LATENCIES:
        return
    path = "reports/latency_breakdown.json"
    try:
        with open(path, encoding="utf-8") as f:
            timings = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        timings = {}
    for key in ("search_ms", "rerank_ms", "generation_ms", "total_ms"):
        values = [row[key] for row in _QUERY_LATENCIES]
        timings[f"query_{key}_avg"] = round(sum(values) / len(values), 3)
        timings[f"query_{key}_p95"] = round(sorted(values)[max(0, int(len(values) * 0.95) - 1)], 3)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
