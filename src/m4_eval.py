from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import asyncio
import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH
from config import OPENAI_API_KEY


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str],
                   use_live: bool = False) -> dict:
    """Run RAGAS evaluation."""
    if not use_live or not OPENAI_API_KEY:
        return _heuristic_eval(questions, answers, contexts, ground_truths)

    try:
        return asyncio.run(_evaluate_ragas_live(
            questions, answers, contexts, ground_truths
        ))
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed, using deterministic heuristic eval: {e}")
        return _heuristic_eval(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating or using facts not grounded in retrieved context",
            "Tighten the answer prompt, lower temperature, and require citations from retrieved chunks",
        ),
        "context_recall": (
            "Missing relevant chunks before generation",
            "Improve chunking, add query expansion, or increase BM25/dense candidate limits",
        ),
        "context_precision": (
            "Retrieved set contains too many irrelevant chunks",
            "Add stronger reranking, metadata filters, or stricter hybrid top-k selection",
        ),
        "answer_relevancy": (
            "Answer does not directly address the user question",
            "Improve prompt template and add question-aware answer validation",
        ),
    }
    rows = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg = _mean(metrics.values())
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        rows.append({
            "question": item.question,
            "expected": item.ground_truth,
            "got": item.answer,
            "worst_metric": worst_metric,
            "score": round(float(metrics[worst_metric]), 4),
            "avg_score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": _error_tree(item, worst_metric),
        })
    return sorted(rows, key=lambda row: row["avg_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


def _heuristic_eval(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = "\n\n".join(ctxs)
        faithfulness_score = _overlap(answer, context_text)
        answer_score = max(_overlap(ground_truth, answer), _overlap(question, answer))
        precision_scores = [_overlap(ground_truth, ctx) for ctx in ctxs] or [0.0]
        precision = max(precision_scores)
        recall = _overlap(ground_truth, context_text)
        per_question.append(EvalResult(
            question=question,
            answer=answer,
            contexts=ctxs,
            ground_truth=ground_truth,
            faithfulness=round(faithfulness_score, 4),
            answer_relevancy=round(answer_score, 4),
            context_precision=round(precision, 4),
            context_recall=round(recall, 4),
        ))
    return {
        "faithfulness": _mean([item.faithfulness for item in per_question]),
        "answer_relevancy": _mean([item.answer_relevancy for item in per_question]),
        "context_precision": _mean([item.context_precision for item in per_question]),
        "context_recall": _mean([item.context_recall for item in per_question]),
        "per_question": per_question,
    }


async def _evaluate_ragas_live(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client, temperature=0)
    embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")
    faithfulness_metric = Faithfulness(llm=llm)
    relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
    precision_metric = ContextPrecisionWithReference(llm=llm)
    recall_metric = ContextRecall(llm=llm)
    semaphore = asyncio.Semaphore(3)

    async def score_one(question, answer, retrieved, reference):
        async with semaphore:
            faithfulness, relevancy, precision, recall = await asyncio.gather(
                faithfulness_metric.ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=retrieved,
                ),
                relevancy_metric.ascore(
                    user_input=question,
                    response=answer,
                ),
                precision_metric.ascore(
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=retrieved,
                ),
                recall_metric.ascore(
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=retrieved,
                ),
            )
            return EvalResult(
                question=question,
                answer=answer,
                contexts=retrieved,
                ground_truth=reference,
                faithfulness=_safe_float(faithfulness.value),
                answer_relevancy=_safe_float(relevancy.value),
                context_precision=_safe_float(precision.value),
                context_recall=_safe_float(recall.value),
            )

    try:
        per_question = await asyncio.gather(*[
            score_one(question, answer, retrieved, reference)
            for question, answer, retrieved, reference in zip(
                questions, answers, contexts, ground_truths
            )
        ])
    finally:
        await client.close()

    return {
        "faithfulness": _mean([item.faithfulness for item in per_question]),
        "answer_relevancy": _mean([item.answer_relevancy for item in per_question]),
        "context_precision": _mean([item.context_precision for item in per_question]),
        "context_recall": _mean([item.context_recall for item in per_question]),
        "per_question": per_question,
    }


def _tokens(text: str) -> set[str]:
    import re

    stop = {"và", "là", "của", "cho", "được", "theo", "khi", "có", "không", "một", "các", "trong"}
    return {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in stop
    }


def _overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _mean(values) -> float:
    values = list(values)
    return round(sum(float(v) for v in values) / max(1, len(values)), 4)


def _safe_float(value) -> float:
    try:
        if value != value:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _error_tree(item: EvalResult, worst_metric: str) -> str:
    context_ok = item.context_recall >= 0.7
    output_ok = item.faithfulness >= 0.7 and item.answer_relevancy >= 0.7
    query_ok = item.context_precision >= 0.7
    return (
        f"Output đúng? {'Có' if output_ok else 'Không'} → "
        f"Context đúng? {'Có' if context_ok else 'Không'} → "
        f"Query/Rerank OK? {'Có' if query_ok else 'Không'} → "
        f"Fix ưu tiên: {worst_metric}"
    )


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
