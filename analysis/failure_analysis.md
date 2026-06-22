# Failure Analysis - Lab 18: Production RAG

**Hình thức:** Cá nhân
**Thành viên:** Ngô Anh Tuấn
**Nguồn số liệu:** Production run bằng Qdrant, OpenAI và RAGAS 0.4

## RAGAS Scores

| Metric | Naive baseline | Production | Delta |
|---|---:|---:|---:|
| Faithfulness | 0.7071 | **0.8881** | **+0.1810** |
| Answer Relevancy | 0.4330 | **0.5402** | **+0.1072** |
| Context Precision | 0.9042 | **0.9417** | **+0.0375** |
| Context Recall | 0.8250 | **0.9500** | **+0.1250** |

## Bottom-5 Failures

### 1. Laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai
  phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt, CNTT xác nhận cấu hình và đính kèm ít nhất
  ba báo giá.
- **Got:** Trả đúng Director và xác nhận kỹ thuật, nhưng thiếu yêu cầu ba báo giá.
- **Worst metric:** `context_precision = 0.3333`.
- **Diagnosis:** Query có ba intent: ngưỡng phê duyệt, xác nhận CNTT và chứng
  từ mua sắm. Candidate set có đúng tài liệu mua sắm nhưng vẫn chứa parent
  nghỉ phép do các từ phổ biến như “30” và “phê duyệt”.
- **Error Tree:** Output đầy đủ? Không -> Context đúng? Có một phần -> Query/
  rerank tốt? Chưa -> Fix tại retrieval diversity và answer completeness.
- **Fix:** Filter/rerank theo metadata `category=procurement|it`, áp dụng MMR
  theo source, đồng thời kiểm tra sau generation rằng mỗi vế hỏi đã có câu trả lời.

### 2. Senior 9 năm: phép năm và khoảng lương

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu
  ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép; lương Senior 20-35 triệu VNĐ/tháng.
- **Got:** Tính đúng 18 ngày nhưng không đưa ra khoảng lương.
- **Worst metric:** `context_recall = 0.5000`.
- **Diagnosis:** Đây là multi-hop giữa chính sách phép năm và bảng lương.
  Query decomposition tìm được nhánh lương, nhưng global rerank vẫn ưu tiên
  nhiều tài liệu nghỉ phép có lexical overlap cao.
- **Error Tree:** Output đầy đủ? Không -> Context đủ hai nguồn? Không ->
  Sub-query đúng? Có -> Fix tại bước hợp nhất/rerank.
- **Fix:** Dành tối thiểu một context slot cho mỗi sub-query trước khi global
  rerank, rồi deduplicate theo source. Đây là intent-balanced retrieval thay
  vì chỉ tăng top-k.

### 3. Phạt tạm ứng quá hạn

- **Question:** Tạm ứng 15 triệu, sau 20 ngày mới thanh toán, bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; pro-rata khoảng 50.000 VNĐ.
- **Got:** Áp phí trọn tháng và trả 300.000 VNĐ.
- **Worst metric:** `answer_relevancy = 0.4049`.
- **Diagnosis:** Retrieval đúng chính sách 15 ngày và 2%/tháng, nhưng văn bản
  không định nghĩa rõ quy tắc làm tròn/pro-rata. LLM tự chọn cách tính trọn tháng.
- **Error Tree:** Output đúng? Không -> Context đúng? Có nhưng thiếu công thức
  ngày -> Retrieval đúng? Có -> Fix tại business rule/calculation tool.
- **Fix:** Bổ sung quy tắc pro-rata vào policy hoặc chuyển phép tính sang hàm
  deterministic: `principal * monthly_rate * overdue_days / 30`; LLM chỉ giải thích kết quả.

### 4. Bảo hiểm PVI trong thử việc

- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected/Got:** Đều trả lời không; chỉ tham gia bảo hiểm xã hội bắt buộc.
- **Worst metric:** `answer_relevancy = 0.4475`.
- **Diagnosis:** Câu trả lời đúng nội dung và grounded, nhưng Answer Relevancy
  của RAGAS dùng reverse-question similarity nên có độ biến thiên với câu
  yes/no tiếng Việt ngắn.
- **Error Tree:** Output đúng? Có -> Context đúng? Có -> Retrieval đúng? Có ->
  Kiểm tra evaluator và format trả lời.
- **Fix:** Dùng template “Có/Không + chủ thể + quyền lợi”, tăng `strictness`
  của evaluator và báo cáo mean/std qua nhiều lần chạy để giảm variance.

### 5. Nghỉ phép năm trong thử việc

- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected/Got:** Đều trả lời không; nếu cần nghỉ phải xin nghỉ không lương
  và được trưởng phòng phê duyệt.
- **Worst metric:** `answer_relevancy = 0.4606`.
- **Diagnosis:** Tương tự failure 4, factual answer đúng nhưng metric relevancy
  thấp. Context precision của câu này cũng bị ảnh hưởng bởi các chính sách
  nghỉ khác cùng chia sẻ từ khóa.
- **Error Tree:** Output đúng? Có -> Context đúng? Có -> Candidate set còn
  nhiễu? Có -> Fix retrieval filter và evaluator stability.
- **Fix:** Lọc theo metadata `employment_status=probation`, ưu tiên parent từ
  chính sách thử việc, và dùng thêm exact semantic similarity với reference
  như một secondary evaluation signal.

## Kết luận chẩn đoán

Failure không tập trung ở một module duy nhất:

- Retrieval còn thiếu intent balancing cho câu multi-hop.
- Generation cần một calculator/rule engine cho phép tính tài chính.
- Answer relevancy cần đánh giá ổn định hơn cho câu yes/no tiếng Việt.

Ưu tiên tiếp theo là intent-balanced context selection, sau đó mới thay lexical
fallback bằng CrossEncoder khi runtime có wheel tương thích Python 3.14.
