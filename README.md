# Wikipedia RfA · Khai phá dữ liệu Web

Dự án phân tích các cuộc thảo luận và biểu quyết Wikipedia Requests for Adminship
(RfA). Mục tiêu là nghiên cứu thái độ ủng hộ, phản đối và trung lập trong bình luận,
phân tích mạng lưới biểu quyết có dấu và dự đoán kết quả bầu chọn từ dữ liệu ban đầu.

## Dữ liệu trong repository

Repository cung cấp ba file dữ liệu cuối cho các tác vụ downstream:

| File | Vai trò |
|---|---|
| `data/stance_model.csv` | Dữ liệu bình luận dùng cho mô hình phân loại stance |
| `data/election_model.csv` | Dữ liệu vote có timestamp dùng cho dự đoán kết quả election |
| `data/rfa_modeling.csv.summary.json` | Số mẫu, cột dữ liệu, trạng thái text và checksum nguồn |

Hai file CSV đã được làm sạch, nối dữ liệu và tạo `model_election_id`.
Mỗi dòng là một lượt vote/bình luận trong một election.

## Các trường chính

| Trường | Ý nghĩa |
|---|---|
| `textid` | ID duy nhất của một vote/comment trong dữ liệu SNAP |
| `model_election_id` | ID của election sau khi thống nhất dữ liệu |
| `election_id_source` | Election ID lấy từ graph hoặc suy ra bằng quy tắc 7 ngày |
| `graph_election_id` | ID election trong mạng graph; có thể rỗng khi không nối được trực tiếp |
| `timestamp` | Thời điểm vote theo UTC |
| `timestamp_available` | Vote có timestamp dùng được hay không |
| `vote_order` | Thứ tự vote trong election |
| `voter` | Người bỏ phiếu |
| `candidate` | Ứng viên |
| `stance` | `1`: ủng hộ, `0`: trung lập, `-1`: phản đối |
| `outcome` | `1`: được chấp thuận, `-1`: bị từ chối |
| `txt_clean` | Bình luận sau khi loại vote label, chữ ký và wikitext |
| `text_status` | `ready`, `needs_review`, `empty` hoặc `symbols_only` |
| `text_flags` | Các thao tác hoặc cảnh báo trong quá trình làm sạch text |
| `eligible_for_stance_model` | Row có comment sạch đủ điều kiện cho stance model |
| `eligible_for_election_model` | Row có timestamp, election ID và outcome để phân tích early votes |

## Tập dữ liệu stance

Sử dụng `data/stance_model.csv`.

- Số dòng: 170.540.
- Input: `txt_clean`.
- Target: `stance`.
- Mọi dòng đều có `eligible_for_stance_model = True`.
- Mỗi dòng là một bình luận ở trạng thái `ready`.

Các nhãn stance:

```text
 1 = Support / Ủng hộ
 0 = Neutral / Trung lập
-1 = Oppose / Phản đối
```

Pipeline baseline đề xuất:

```text
txt_clean -> TF-IDF -> Logistic Regression hoặc Linear SVM -> stance
```

Metric chính là macro-F1 ba lớp. Không nên chỉ dùng accuracy vì dữ liệu stance mất
cân bằng, đặc biệt lớp Neutral.

## Tập dữ liệu election

Sử dụng `data/election_model.csv`.

- Số dòng: 188.904.
- Mọi dòng đều có timestamp, election ID và outcome.
- Mỗi dòng có `eligible_for_election_model = True`.
- Dùng `vote_order` để lấy các cửa sổ 5, 10 và 20 votes đầu.

Để dự đoán outcome:

1. Nhóm dữ liệu theo `model_election_id`.
2. Sắp theo `vote_order` hoặc `timestamp`.
3. Chỉ lấy prefix đã quan sát tại thời điểm dự đoán.
4. Tạo feature từ text, signed votes hoặc cả hai.
5. Dùng `outcome` làm target.

Các mô hình cần so sánh:

- Text-only.
- Graph/vote-only.
- Combined.
- Majority-class baseline.
- Early signed-support-ratio baseline.

## Chia train/validation/test theo thời gian

Không chia ngẫu nhiên vì model thực tế chỉ được dùng dữ liệu quá khứ để dự đoán
tương lai.

Thiết lập đề xuất:

```text
Train:      2004-2007
Validation: 2008
Test:       2009-2012
```

Toàn bộ votes và comments thuộc cùng một election phải nằm trong cùng split. Metric
chính cho outcome prediction là macro-F1 hai lớp.

## Tránh rò rỉ dữ liệu

- Không dùng `outcome` làm feature.
- Không giữ nhãn `Support/Oppose/Neutral` trong text input.
- Khi dự đoán sau 5 hoặc 10 votes, không dùng các votes sau đó.
- Không dùng số lượng vote cuối election để xây dựng early-window feature.
- Feature graph chỉ được phép dùng thông tin từ các election trước thời điểm dự đoán.
- Fit TF-IDF, scaler và các bộ tham số chỉ trên train split.

## Election ID

Election ID được thống nhất theo cách sau:

1. Nếu vote nối được với graph qua `textid = record_id`, ưu tiên `graph.election_id`.
2. Truyền graph ID đó cho toàn bộ votes trong cùng election, kể cả neutral votes.
3. Nếu election không có trong graph, dùng key suy luận:

```text
candidate + year + outcome + vote_start
```

`outcome` chỉ được dùng để phân biệt các nhóm election trong dữ liệu lịch sử. Khi mô
hình hóa, nó phải được tách riêng thành target và tuyệt đối không đưa vào feature.

## Giá trị rỗng

Một số trường có thể rỗng nhưng không phải lỗi:

- `graph_election_id` rỗng: election không match trực tiếp với graph.
- `timestamp` rỗng: dữ liệu gốc không có thời gian hợp lệ.
- `txt_clean` rỗng: không có comment usable sau preprocessing.
- `text_flags` rỗng: quá trình làm sạch không phát hiện warning hay thao tác cần ghi.

Trong các file task-specific, dữ liệu đã được lọc bằng cờ `eligible_for_*`, nên không
cần tự động xóa mọi giá trị rỗng.

## Kết quả dữ liệu

Tệp `data/rfa_modeling.csv.summary.json` ghi lại:

- 198.275 rows trước khi tách task.
- 4.120 elections đã thống nhất.
- 170.540 rows cho stance model.
- 188.904 rows cho election model.
- Phân bố trạng thái text.
- SHA-256 của các file nguồn.
- Quy tắc tạo election ID.

## Phạm vi hiện tại

Repository hiện cung cấp dữ liệu đã được chuẩn bị cho modeling. Phần còn lại là:

1. Train stance model.
2. Tạo early-window text features.
3. Tạo graph features có tính đến thời gian.
4. Train và so sánh text-only, graph-only và combined models.
5. Báo cáo macro-F1 và performance theo observation window.
