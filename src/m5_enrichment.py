from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re, hashlib
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "enrichment.json")
_ENRICHMENT_CACHE: dict[str, dict] | None = None


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    sentences = _sentences(text)
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0,
            )
            questions = resp.choices[0].message.content.strip().splitlines()
            return [_clean_question(q) for q in questions if q.strip()][:_safe_n(n_questions)]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    questions = []
    for sentence in _sentences(text)[:_safe_n(n_questions)]:
        subject = sentence.rstrip(".")
        if re.search(r"\bbao nhiêu|mấy|không|cần|phải\b", subject.lower()):
            questions.append(subject + "?")
        else:
            questions.append(f"Thông tin chính của quy định sau là gì: {subject}?")
    return questions


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
                temperature=0,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    prefix = f"Ngữ cảnh: trích từ {document_title}. " if document_title else "Ngữ cảnh: đoạn chính sách nội bộ. "
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Chỉ trả JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            return _parse_json(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    cache_key = hashlib.sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{
  "summary": "tóm tắt 2 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = _parse_json(resp.choices[0].message.content)
            if parsed:
                cache[cache_key] = parsed
                _save_cache(cache)
                return parsed
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")
    fallback = {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Ngữ cảnh: trích từ {source}." if source else "Ngữ cảnh: đoạn chính sách nội bộ.",
        "metadata": _fallback_metadata(text),
    }
    return fallback


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enrichment_parts = []
            if context_line:
                enrichment_parts.append(context_line)
            if summary:
                enrichment_parts.append(f"Tóm tắt: {summary}")
            if questions:
                enrichment_parts.append(
                    "Câu hỏi liên quan: " + " | ".join(str(q) for q in questions)
                )
            enrichment_parts.append(text)
            enriched_text = "\n\n".join(enrichment_parts)
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.replace("\r", "\n")) if len(s.strip()) > 5]


def _clean_question(question: str) -> str:
    question = question.strip().lstrip("0123456789.-) ").strip()
    return question if question.endswith("?") else f"{question}?"


def _safe_n(n: int) -> int:
    return max(1, min(10, int(n)))


def _parse_json(content: str) -> dict:
    content = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL)
    if fenced:
        content = fenced.group(1).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
    return {}


def _fallback_metadata(text: str) -> dict:
    lower = text.lower()
    if any(word in lower for word in ["mật khẩu", "vpn", "malware", "cntt", "mfa"]):
        category = "it"
    elif any(word in lower for word in ["lương", "phụ cấp", "tạm ứng", "chi phí", "bảo hiểm"]):
        category = "finance"
    elif any(word in lower for word in ["nghỉ", "thử việc", "mentor", "đào tạo"]):
        category = "hr"
    else:
        category = "policy"
    entities = sorted(set(re.findall(r"\b[A-ZĐ][\wÀ-ỹ-]{2,}\b", text, flags=re.UNICODE)))[:8]
    topic = summarize_chunk(text)[:120].rstrip(".")
    return {"topic": topic or "general", "entities": entities, "category": category, "language": "vi"}


def _load_cache() -> dict[str, dict]:
    global _ENRICHMENT_CACHE
    if _ENRICHMENT_CACHE is not None:
        return _ENRICHMENT_CACHE
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            _ENRICHMENT_CACHE = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _ENRICHMENT_CACHE = {}
    return _ENRICHMENT_CACHE


def _save_cache(cache: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    tmp_path = f"{_CACHE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp_path, _CACHE_PATH)


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
