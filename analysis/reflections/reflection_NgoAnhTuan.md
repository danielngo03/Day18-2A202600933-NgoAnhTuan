# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Ngô Anh Tuấn  
**Hình thức:** Bài tập cá nhân, phụ trách toàn bộ M1-M5  
**Project cá nhân:** VinUni Career Platform

---

## 1. Mapping bài giảng vào implementation

| Lecture concept | Module | Code cụ thể | Kết quả và nhận xét |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Câu được nhóm bằng cosine similarity; có giới hạn độ dài tối thiểu/tối đa để tránh tạo chunk quá vụn khi ngưỡng similarity cao. Trên 26 tài liệu có text, semantic tạo 59 chunks so với 51 basic chunks; biên chunk bám theo ý nghĩa câu thay vì chỉ theo số ký tự. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Tạo parent 2,048 ký tự và child 256 ký tự. Pipeline index/retrieve child để tăng precision, sau đó dùng `parent_id` trả parent text cho LLM để giữ đủ ngữ cảnh. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Parse Markdown headers, giữ header trong text và lưu `section` trong metadata. Cách này phù hợp với tài liệu chính sách có heading, bảng và danh sách. |
| Sparse retrieval cho từ khóa chính xác | M2 | `segment_vietnamese()`, `BM25Search` | BM25 xử lý tốt số liệu, tên chính sách và thuật ngữ như “MFA”, “PVI”, “nghỉ phép”. Sau word segmentation, dấu `_` được thay bằng khoảng trắng để query và corpus dùng cùng token form. |
| Dense retrieval cho semantic similarity | M2 | `DenseSearch.index()`, `DenseSearch.search()` | Production run dùng OpenAI `text-embedding-3-small` 1,024 chiều, lưu vector trong Qdrant và truy vấn bằng `query_points()`. Dense search bổ sung các kết quả diễn đạt khác từ khóa của câu hỏi. |
| Reciprocal Rank Fusion | M2 | `reciprocal_rank_fusion()` | RRF hợp nhất BM25 và dense theo thứ hạng, không phụ thuộc hai hệ score khác scale. Kết quả cuối có `method="hybrid"`. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Class cấu hình `BAAI/bge-reranker-v2-m3`; runtime Python 3.14 hiện dùng lexical fallback vì thiếu wheel `sentence-transformers`. Pipeline vẫn rerank toàn bộ candidates, cộng lifecycle score và deduplicate parent trước khi lấy top-3. |
| Component-wise evaluation | M4 | `evaluate_ragas()` | RAGAS live đạt faithfulness 0.8881, answer relevancy 0.5402, context precision 0.9417 và context recall 0.9500. Việc tách metric cho thấy retrieval mạnh, còn answer relevancy là hướng cải thiện chính. |
| Diagnostic Error Tree | M4 | `failure_analysis()` | Mỗi câu bottom-N có worst metric, diagnosis, suggested fix và chuỗi kiểm tra “Output → Context → Query/Rerank”. Đây là cách chuyển score thành hành động kỹ thuật cụ thể. |
| Contextual enrichment và HyQA | M5 | `_enrich_single_call()` | Một OpenAI call/chunk sinh summary, hypothesis questions, contextual description và metadata; tất cả được đưa vào `enriched_text` trước embedding. Cache SHA-256 giảm enrichment 114 chunks từ khoảng 457.3 giây ở cold run xuống 0.003 giây ở warm run. |
| Latency/cost observability | Pipeline | `reports/latency_breakdown.json` | Pipeline đo average/p95 cho search, rerank, generation và end-to-end. Run cuối có trung bình 2,110 ms/query và p95 3,203 ms/query, giúp xác định generation là thành phần chiếm latency lớn nhất. |

## 2. Khó khăn và cách giải quyết

### 2.1 Mâu thuẫn giữa nhiều phiên bản chính sách

- **Failure quan sát được:** câu hỏi “Bao lâu phải đổi mật khẩu một lần?” có thể retrieve chính sách v1.0 quy định 90 ngày thay vì v2.0 hiện hành quy định 120 ngày.
- **Quá trình debug:** kiểm tra top retrieved contexts, đối chiếu `source`, rồi đi ngược Error Tree. Query đúng chủ đề nhưng context đứng đầu là tài liệu cũ, nên lỗi nằm ở retrieval/reranking chứ không phải prompt generation.
- **Cách giải quyết:** bổ sung metadata `document_status` và `policy_priority`, giảm điểm tài liệu `superseded`, tăng điểm tài liệu `current`, đồng thời yêu cầu answer prompt giải quyết rõ xung đột phiên bản.
- **Kiến thức rút ra:** semantic similarity không thể thay thế lifecycle metadata. Production RAG phải quản lý hiệu lực tài liệu như một business rule.

### 2.2 Child chunk chính xác nhưng thiếu dữ kiện liền kề

- **Failure quan sát được:** query về malware retrieve đúng câu “không tự ý xử lý”, nhưng child chunk có thể thiếu SLA “báo cáo trong vòng 1 giờ”.
- **Quá trình debug:** context precision tốt nhưng context recall thấp; kiểm tra parent cho thấy dữ kiện còn lại nằm trong cùng section, ngay bên cạnh child được retrieve.
- **Cách giải quyết:** index child để tìm kiếm, lưu `parent_id` và `parent_text`, sau reranking trả parent context cho LLM.
- **Kiến thức rút ra:** retrieval unit và generation context không nhất thiết phải cùng kích thước.

### 2.3 Tương thích runtime Python 3.14

- **Exact error:** `ModuleNotFoundError: No module named 'sentence_transformers'` và `ModuleNotFoundError: No module named 'ragas'`.
- **Quá trình debug:** kiểm tra từng import bằng đúng interpreter Python 3.14, xác định requirements cũ ghim RAGAS/LangChain vào NumPy 1.26 không có wheel phù hợp.
- **Cách giải quyết:** tách môi trường Python 3.14, nâng RAGAS lên nhánh hiện đại, bỏ dependency không sử dụng, đồng thời giữ deterministic fallback để unit test và pipeline không mất hoàn toàn khả năng hoạt động khi provider/model tạm thời unavailable.
- **Kiến thức rút ra:** production pipeline cần pin dependency theo runtime thực tế và có graceful degradation, nhưng report phải ghi rõ run nào dùng model thật và run nào dùng fallback.

## 3. Action plan cho VinUni Career Platform

### Bối cảnh project

VinUni Career Platform kết nối ba nhóm người dùng:

- **Sinh viên:** quản lý hồ sơ/CV, tìm cơ hội, nhận giải thích mức độ phù hợp, ứng tuyển và chuẩn bị phỏng vấn.
- **Doanh nghiệp:** đăng mô tả công việc, tìm ứng viên phù hợp, quản lý các vòng tuyển dụng, phỏng vấn và offer.
- **Nhà trường:** kiểm duyệt doanh nghiệp/cơ hội, theo dõi chất lượng tuyển dụng và quản trị rủi ro AI.

Hệ thống hiện dùng backend FastAPI, frontend Next.js và một AI gateway có nhiều provider cùng offline fallback. Các chức năng AI đã bao gồm trích xuất CV, chuẩn hóa kỹ năng, so khớp CV-JD, trợ lý nghề nghiệp và tìm kiếm kết hợp keyword/vector. Tuy nhiên, retrieval vẫn cần được chuẩn hóa thành một pipeline có evaluation gate, citation, version metadata và quan sát chi phí/latency.

### Vấn đề hiện tại

- CV và JD có cấu trúc khác nhau nhưng chưa có chunking strategy chuyên biệt cho từng loại tài liệu.
- Match theo kỹ năng chính xác cần sparse search, còn mô tả kinh nghiệm tương đương cần dense search.
- Trợ lý nghề nghiệp có thể lấy context đúng chủ đề nhưng thiếu section liên quan hoặc thiếu nguồn trích dẫn.
- Chính sách ứng tuyển, quyền riêng tư và quy trình tuyển dụng có thể thay đổi theo phiên bản.
- Chưa có bộ benchmark ổn định để ngăn chất lượng AI giảm sau khi đổi model hoặc prompt.

### Kế hoạch áp dụng

1. [ ] **Chunking:** dùng structure-aware + hierarchical chunking. CV chia theo Summary, Education, Experience, Skills, Projects; JD chia theo Responsibilities, Requirements, Benefits; policy chia theo heading và version.
2. [ ] **Hybrid search:** dùng BM25 cho skill, chứng chỉ, chức danh và con số; dùng dense search cho kinh nghiệm tương đương; hợp nhất bằng RRF.
3. [ ] **Reranking:** rerank top-30 xuống top-8 trước khi tạo giải thích CV-JD hoặc trả lời career assistant. Áp dụng metadata filter theo tenant, privacy level và document status trước reranking.
4. [ ] **Evaluation:** xây test set tối thiểu 100 câu thuộc bốn nhóm: job recommendation, CV improvement, application policy và interview/offer workflow. Dùng RAGAS cho RAG answer và custom metrics cho match-score calibration.
5. [ ] **Enrichment:** sinh summary, hypothetical questions và metadata `document_type`, `job_id`, `organization_id`, `version`, `effective_date`, `privacy_level`, `current|superseded`.
6. [ ] **Safety:** luôn mask PII trước khi index CV, scope retrieval theo quyền người dùng và ghi audit trail cho các AI decision quan trọng.
7. [ ] **Observability:** lưu retrieval trace, source citation, latency từng bước, token usage, user feedback và failure category.

### Timeline

| Thời gian | Deliverable | Tiêu chí hoàn thành |
|---|---|---|
| Tuần 1 | Metadata schema + chunking theo CV/JD/policy | Unit tests cho section, parent-child và privacy metadata |
| Tuần 2 | BM25 + dense + RRF + Qdrant | Recall@10 và MRR được đo trên benchmark |
| Tuần 3 | Cross-encoder rerank + parent expansion | Context precision tăng, p95 latency nằm trong budget |
| Tuần 4 | Career assistant có citation | Mọi claim quan trọng trỏ về source chunk/document |
| Tuần 5 | RAGAS/custom evaluation trong CI | Build bị chặn nếu metric giảm quá ngưỡng cho phép |
| Tuần 6 | AI governance dashboard | Theo dõi quality, latency, cost, feedback và privacy events |

## 4. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Minh chứng |
|---|---:|---|
| Hiểu bài giảng | 5 | Mapping đủ M1-M5 và giải thích trade-off |
| Code quality | 5 | Typed dataclasses, fallback, cache, report và tests |
| Problem solving | 5 | Diagnostic Tree dẫn tới fix versioning và parent expansion |
| Khả năng áp dụng project | 5 | Plan có use case, safety, metric và timeline cụ thể |
