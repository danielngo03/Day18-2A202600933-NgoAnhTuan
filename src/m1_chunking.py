from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re, math, hashlib
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  ⚠️  Bỏ qua PDF: chưa cài pypdf.")
        return ""

    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception as exc:
        print(f"  ⚠️  Không đọc được {os.path.basename(path)}: {exc}")
        return ""


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0})]

    embeddings = _semantic_embeddings(sentences)
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        current_len = len("\n\n".join(groups[-1]))
        should_split = sim < threshold and current_len >= 300
        should_force_split = current_len >= 900
        if (should_split or should_force_split) and "\n\n".join(groups[-1]).strip():
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(
            text="\n\n".join(group).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": i},
        )
        for i, group in enumerate(groups)
        if "\n\n".join(group).strip()
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > parent_size:
            _append_parent(parents, current, metadata)
            current = ""
        if len(para) > parent_size:
            if current:
                _append_parent(parents, current, metadata)
                current = ""
            for piece in _split_by_size(para, parent_size):
                _append_parent(parents, piece, metadata)
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        _append_parent(parents, current, metadata)

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        child_texts = _split_by_size(parent.text, child_size)
        for child_index, child_text in enumerate(child_texts):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "child_index": child_index,
                    "parent_id": pid,
                },
                parent_id=pid,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_header = metadata.get("source", "Document")
    current_lines: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+\S", line):
            if current_lines:
                _append_section(chunks, current_header, current_lines, metadata)
            current_header = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        _append_section(chunks, current_header, current_lines, metadata)
    if not chunks and text.strip():
        chunks.append(Chunk(text.strip(), {**metadata, "section": current_header, "strategy": "structure", "chunk_index": 0}))
    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
    return [part.strip() for part in parts if part.strip()]


def _semantic_embeddings(sentences: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return [list(map(float, emb)) for emb in model.encode(sentences)]
    except Exception:
        return [_hashed_bow(sentence) for sentence in sentences]


def _hashed_bow(text: str, dim: int = 384) -> list[float]:
    vector = [0.0] * dim
    for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
        vector[_stable_bucket(token, dim)] += 1.0
    return vector


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % dim


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / max(1e-9, norm_a * norm_b)


def _append_parent(parents: list[Chunk], text: str, metadata: dict) -> None:
    pid = f"parent_{len(parents)}"
    parents.append(Chunk(
        text=text.strip(),
        metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
    ))


def _split_by_size(text: str, size: int) -> list[str]:
    units = _split_sentences(text)
    if not units:
        units = [text.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(unit[i:i + size].strip() for i in range(0, len(unit), size) if unit[i:i + size].strip())
            continue
        if current and len(current) + len(unit) + 1 > size:
            chunks.append(current.strip())
            current = unit
        else:
            current = f"{current} {unit}".strip() if current else unit
    if current:
        chunks.append(current.strip())
    return chunks


def _append_section(chunks: list[Chunk], header: str, lines: list[str], metadata: dict) -> None:
    body = "\n".join(lines).strip()
    if body:
        section = header.lstrip("#").strip()
        chunks.append(Chunk(
            text=body,
            metadata={**metadata, "section": section, "strategy": "structure", "chunk_index": len(chunks)},
        ))


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
