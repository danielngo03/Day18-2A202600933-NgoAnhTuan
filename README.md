# Lab 18: Production RAG Pipeline

**AICB-P2T3 - Production RAG**<br>
**Sinh viên:** Ngô Anh Tuấn<br>
**Hình thức:** Cá nhân

## Tổng quan

Đây là phiên bản hoàn chỉnh của Lab 18, xây dựng một hệ thống RAG tiếng Việt
theo hướng production trên corpus chính sách nội bộ. Hệ thống triển khai đủ
5 module của đề bài, chạy end-to-end với Qdrant và OpenAI, đánh giá bằng
RAGAS, phân tích lỗi và ghi nhận latency.

```text
Documents
   |
   v
M1 Hierarchical Chunking
   |
   v
M5 Combined Enrichment + SHA-256 Cache
   |
   v
M2 BM25 + Dense Search + Reciprocal Rank Fusion
   |
   v
Query Decomposition + M3 Reranking + Parent Expansion
   |
   v
GPT-4o-mini Answer Generation
   |
   v
M4 RAGAS Evaluation + Failure Analysis
```

Pipeline production được so sánh với baseline gồm paragraph chunking,
dense-only retrieval, không enrichment và không reranking.

## Kết quả chính

Toàn bộ **37/37 automated tests** đã pass trên Python 3.14.5.

| Metric | Naive baseline | Production | Delta |
|---|---:|---:|---:|
| Faithfulness | 0.7071 | **0.8881** | **+0.1810** |
| Answer Relevancy | 0.4330 | **0.5402** | **+0.1072** |
| Context Precision | 0.9042 | **0.9417** | **+0.0375** |
| Context Recall | 0.8250 | **0.9500** | **+0.1250** |

Ba metric production đạt từ 0.70 trở lên; Faithfulness đạt trên 0.85.
Các số liệu trên là snapshot của lần chạy đã lưu trong `reports/`. Kết quả
RAGAS có thể dao động nhẹ khi chạy lại do sử dụng LLM làm evaluator.

Latency của warm-cache run:

| Bước | Average | P95 |
|---|---:|---:|
| Chunking | 0.108 s | - |
| Enrichment từ cache | 0.003 s | - |
| Dense indexing | 4.151 s | - |
| Search | 567.247 ms | 1425.671 ms |
| Reranking | 2.171 ms | 4.686 ms |
| Answer generation | 1540.700 ms | 2246.922 ms |
| End-to-end/query | **2110.296 ms** | **3202.747 ms** |

## Các module đã triển khai

### M1 - Advanced Chunking

- `chunk_semantic()`: nhóm câu theo cosine similarity, có deterministic
  fallback khi model embedding cục bộ không khả dụng.
- `chunk_hierarchical()`: tạo parent chunk và child chunk; retrieve bằng
  child rồi mở rộng về parent để giữ đầy đủ ngữ cảnh.
- `chunk_structure_aware()`: tách Markdown theo heading và lưu section trong
  metadata.
- `load_documents()`: đọc Markdown và trích xuất text từ PDF có text layer.

### M2 - Hybrid Search

- Tách từ tiếng Việt bằng `underthesea`.
- Sparse retrieval bằng BM25.
- Dense retrieval bằng Qdrant.
- Hợp nhất hai ranking bằng Reciprocal Rank Fusion (RRF).
- OpenAI `text-embedding-3-small` được dùng làm dense encoder trong môi
  trường Python 3.14 hiện tại; code vẫn hỗ trợ `BAAI/bge-m3` khi
  `sentence-transformers` khả dụng.
- Có in-memory fallback để unit test không phụ thuộc dịch vụ bên ngoài.

### M3 - Reranking

- Cấu hình CrossEncoder `BAAI/bge-reranker-v2-m3`.
- Rerank toàn bộ fused candidates trước khi loại các child trùng parent.
- Có lexical fallback ổn định khi CrossEncoder không tải được trên runtime.
- Điều chỉnh score theo metadata vòng đời tài liệu để ưu tiên chính sách
  hiện hành và hạ hạng phiên bản đã bị thay thế.

### M4 - Evaluation

- Đánh giá 20 câu hỏi bằng RAGAS 0.4 với bốn metric:
  Faithfulness, Answer Relevancy, Context Precision và Context Recall.
- Có deterministic evaluation fallback khi live evaluator không khả dụng.
- Tự động chọn các câu có điểm thấp, xác định metric yếu nhất, ánh xạ Error
  Tree và đề xuất hướng sửa.

### M5 - Enrichment

- Combined mode chỉ dùng **một OpenAI call trên mỗi chunk** để tạo summary,
  hypothetical questions (HyQA), contextual description và metadata.
- Ghép đầy đủ enrichment vào text trước khi tạo embedding.
- Cache theo SHA-256 giúp không gọi lại API cho nội dung chưa thay đổi.
- Fallback cục bộ giữ pipeline hoạt động khi không có API key.

### Production orchestration

- Nhận biết metadata `current`, `superseded` và `unspecified`.
- Tách câu hỏi nhiều vế nhưng vẫn giữ entity, acronym và con số làm anchor.
- Truy hồi child chính xác, rerank parent context, rồi deduplicate theo
  parent để tăng source diversity.
- Sinh câu trả lời ngắn, độc lập, đủ các ý và chỉ dựa trên context.
- Ghi average/P95 latency cho search, rerank, generation và toàn query.

## Yêu cầu môi trường

- Python **3.14**; lần chạy chính thức dùng Python 3.14.5.
- Docker Desktop hoặc Docker Engine có Compose.
- OpenAI API key cho embeddings, enrichment, answer generation và live
  RAGAS evaluation.

Qdrant sử dụng:

- REST API: `localhost:6333`
- gRPC API: `localhost:6334`
- Collections: `lab18_naive` và `lab18_production`

## Cài đặt

Từ thư mục `Lab18`:

```bash
python3.14 -m venv .venv314
source .venv314/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tạo file môi trường:

```bash
cp .env.example .env
```

Sau đó điền key vào `.env`:

```dotenv
OPENAI_API_KEY=sk-...
```

Không commit `.env`; file này đã được khai báo trong `.gitignore`.

## Khởi động Qdrant

```bash
docker compose up -d
docker compose ps
curl http://localhost:6333/healthz
```

Kết quả health check mong đợi:

```text
healthz check passed
```

Xem log hoặc dừng dịch vụ:

```bash
docker compose logs -f qdrant
docker compose down
```

Volume `qdrant_data` giữ lại index giữa các lần khởi động. Dùng
`docker compose down -v` chỉ khi cần xóa toàn bộ dữ liệu Qdrant.

## Chạy dự án

Kích hoạt môi trường trước:

```bash
source .venv314/bin/activate
```

Chạy baseline riêng:

```bash
python naive_baseline.py
```

Chạy production pipeline riêng:

```bash
python src/pipeline.py
```

Chạy baseline, production và in bảng so sánh:

```bash
python main.py
```

Lần chạy đầu có thể lâu vì M5 phải enrichment toàn bộ chunk. Những lần sau
sẽ đọc `.cache/enrichment.json`. Pipeline có thể gọi OpenAI nhiều lần và phát
sinh chi phí API; không nên xóa cache nếu nội dung corpus không thay đổi.

## Kiểm thử

Chạy toàn bộ test:

```bash
python -m pytest tests/ -v
```

Kết quả hiện tại:

| Module | Tests |
|---|---:|
| M1 Chunking | 13/13 |
| M2 Hybrid Search | 5/5 |
| M3 Reranking | 5/5 |
| M4 Evaluation | 4/4 |
| M5 Enrichment | 10/10 |
| **Tổng** | **37/37** |

Kiểm tra đầy đủ artifact trước khi nộp:

```bash
python check_lab.py
```

Có thể kiểm tra thêm theo rubric:

```bash
ruff check src/
grep -r "# TODO" src/m*.py
```

## Dữ liệu và cấu trúc thư mục

Corpus gồm 25 tài liệu Markdown, 3 tài liệu PDF và test set gồm 20 câu hỏi
tiếng Việt về chính sách nhân sự, lương thưởng, bảo mật, vận hành và tuân thủ.

```text
Lab18/
├── README.md
├── ASSIGNMENT.md
├── RUBRIC.md
├── main.py
├── naive_baseline.py
├── check_lab.py
├── config.py
├── docker-compose.yml
├── requirements.txt
├── test_set.json
├── data/
│   ├── *.md
│   └── *.pdf
├── src/
│   ├── m1_chunking.py
│   ├── m2_search.py
│   ├── m3_rerank.py
│   ├── m4_eval.py
│   ├── m5_enrichment.py
│   └── pipeline.py
├── tests/
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
├── reports/
│   ├── naive_baseline_report.json
│   ├── ragas_report.json
│   └── latency_breakdown.json
└── analysis/
    ├── failure_analysis.md
    ├── group_report.md
    └── reflections/
        └── reflection_NgoAnhTuan.md
```

## Deliverables

- Source code hoàn chỉnh cho M1-M5 và production pipeline.
- `ragas_report.json` tại root theo yêu cầu nộp bài.
- Báo cáo baseline, production và latency trong `reports/`.
- Bottom-5 failure analysis kèm diagnosis, suggested fix và Error Tree.
- Group report cho nhóm một thành viên.
- Reflection cá nhân gồm lecture mapping, quá trình debug và action plan áp
  dụng Production RAG vào hệ thống kết nối sinh viên, nhà trường và doanh
  nghiệp.

Phân tích chi tiết và bằng chứng rubric nằm tại:

- `analysis/group_report.md`
- `analysis/failure_analysis.md`
- `analysis/reflections/reflection_NgoAnhTuan.md`
