# Group Report - Lab 18: Production RAG

**Hình thức:** Cá nhân
**Ngày:** 22/06/2026
**Thành viên:** Ngô Anh Tuấn

## Phân công và kiểm thử

| Thành viên | Phạm vi phụ trách | Kết quả |
|---|---|---:|
| Ngô Anh Tuấn | M1 Chunking | 13/13 tests pass |
| Ngô Anh Tuấn | M2 Hybrid Search | 5/5 tests pass |
| Ngô Anh Tuấn | M3 Reranking | 5/5 tests pass |
| Ngô Anh Tuấn | M4 RAGAS Evaluation | 4/4 tests pass |
| Ngô Anh Tuấn | M5 Enrichment | 10/10 tests pass |
| **Tổng** | Toàn bộ pipeline | **37/37 tests pass** |

Môi trường chạy chính thức: Python 3.14.5, Qdrant Docker tại
`localhost:6333`, OpenAI embeddings/generation và RAGAS 0.4.

## Kết quả Evaluation

| Metric | Naive baseline | Production | Delta |
|---|---:|---:|---:|
| Faithfulness | 0.7071 | **0.8881** | **+0.1810** |
| Answer Relevancy | 0.4330 | **0.5402** | **+0.1072** |
| Context Precision | 0.9042 | **0.9417** | **+0.0375** |
| Context Recall | 0.8250 | **0.9500** | **+0.1250** |

Production cải thiện cả bốn metrics. Ba metrics đạt trên 0.70 nên đáp ứng
mức tối đa 10/10 của mục RAGAS trong rubric. Faithfulness vượt 0.85, đủ điều
kiện bonus tương ứng. Answer relevancy vẫn là metric thấp nhất; nhiều câu trả
lời đúng và trực tiếp vẫn nhận điểm thấp, vì vậy đây là hướng cần kiểm định
thêm thay vì che giấu bằng heuristic.

## Thiết kế Production RAG

1. **Chunking:** hierarchical child retrieval + parent context expansion;
   semantic và structure-aware chunking được triển khai độc lập để so sánh.
2. **Enrichment:** một OpenAI call/chunk sinh summary, HyQA, contextual
   description và metadata; cache SHA-256 giúp warm run không gọi lại API.
3. **Retrieval:** BM25 tiếng Việt + OpenAI dense embeddings lưu trong Qdrant,
   hợp nhất bằng Reciprocal Rank Fusion.
4. **Multi-part query:** tách các vế hỏi và giữ entity/acronym/con số làm
   anchor; hợp nhất candidates trước rerank.
5. **Reranking/context:** rerank toàn bộ fused candidates, sau đó mới loại
   child trùng parent để một tài liệu không chiếm hết context window.
6. **Generation/evaluation:** GPT-4o-mini ở temperature 0; RAGAS chấm
   faithfulness, answer relevancy, context precision và context recall.

## Key Findings

- Parent expansion và query decomposition tạo mức tăng lớn nhất ở context
  recall: `0.8250 -> 0.9500`.
- Metadata `current/superseded` là bắt buộc với các chính sách có phiên bản;
  semantic similarity đơn thuần không đủ để loại chính sách cũ.
- HyQA và summary chỉ có tác dụng retrieval khi thực sự được đưa vào
  `enriched_text` trước bước embedding.
- Câu hỏi nhiều vế cần bảo toàn diversity theo parent/source. Cắt top-k trước
  deduplication có thể làm mất tài liệu thứ hai dù retriever đã tìm thấy nó.

## Latency Breakdown

| Bước | Average | P95 |
|---|---:|---:|
| Chunking | 0.108 s | - |
| Enrichment warm cache | 0.003 s | - |
| Dense indexing vào Qdrant | 4.151 s | - |
| Search | 567.247 ms | 1425.671 ms |
| Rerank fallback | 2.171 ms | 4.686 ms |
| LLM generation | 1540.700 ms | 2246.922 ms |
| End-to-end/query | **2110.296 ms** | **3202.747 ms** |

Cold run enrichment 114 chunks mất khoảng 457.3 giây; warm cache giảm xuống
0.003 giây. Điều này cho thấy enrichment nên chạy khi ingest/update tài liệu,
không chạy trong request path.
