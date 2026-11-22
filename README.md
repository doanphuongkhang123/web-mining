# VoteLens · Web Mining

VoteLens chuẩn bị dữ liệu cho bài toán dự đoán kết quả ứng cử quản trị viên Wikipedia sau 10 phiếu đầu quan sát được. Pipeline thực hiện EDA cấp phiếu, đối chiếu từng lần ứng cử, chuẩn bị đầu vào cho nhánh phiếu, văn bản và đồ thị, rồi tạo ma trận TF-IDF riêng cho outcome và stance.

## Chạy toàn bộ pipeline

Dùng Python 3.12. Từ thư mục gốc của repo:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m votelens
```

Lần đầu chạy, script lấy `wiki-RfA.txt.gz` từ Stanford SNAP, xác minh SHA-256 và lưu ở `data/`. Có thể truyền bản sao đã tải bằng `--data /path/to/wiki-RfA.txt.gz`. Repo có snapshot bằng chứng đối chiếu Wikipedia, gồm mã trang, mã revision và ID bản ghi khớp theo phút, để tái tạo tập mẫu đã kiểm tra. Dữ liệu SNAP gốc và kết quả chạy lưu cục bộ. [Stanford SNAP](https://snap.stanford.edu/data/wiki-RfA.html) · [Danh mục lưu trữ RfA của Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Requests_for_adminship_by_year)

Có thể chạy riêng từng phần:

```sh
python -m votelens.eda
python -m votelens.preprocess
python -m votelens.vectorize
```

Dùng `--help` để xem các tùy chọn dữ liệu và thư mục đầu ra. Khi cần thu thập lại snapshot, chạy `python -m votelens.archive --output data/archive_evidence.json.gz --cache data/archive_cache` rồi truyền `--evidence data/archive_evidence.json.gz` cho pipeline. Snapshot mới có thể thay đổi theo revision Wikipedia.

## Đầu ra

| Đường dẫn | Nội dung |
|---|---|
| `outputs/eda/report.html` | EDA cấp phiếu với phân bố phiếu, dữ liệu thiếu, hoạt động theo năm, đồ thị và từ vựng. Bảng CSV và hình PNG nằm cùng thư mục. |
| `outputs/processed/summary.json` | Thống kê chất lượng dữ liệu, số mẫu đủ điều kiện và lý do loại. |
| `outputs/processed/cohort_eda.png` | Độ phủ theo năm, số mẫu ở các mốc quan sát và cân bằng lớp theo tập. |
| `outputs/processed/votes.csv.gz` | Một dòng mỗi bản ghi SNAP, gồm ngày gốc, ngày đã đọc, văn bản gốc, văn bản đã làm sạch và cờ kiểm tra. |
| `outputs/processed/elections.csv` | Một dòng mỗi lần ứng cử đã ghép, có mã trang, revision, giờ đóng nếu đọc được và trạng thái ở từng mốc. |
| `outputs/processed/candidate_run_audit.csv` | Kiểm tra mọi cụm liên tiếp trong SNAP và lý do loại theo mốc. |
| `outputs/processed/prefix_votes.csv.gz` | 10 phiếu đầu của mỗi mẫu outcome, sắp theo thời gian. |
| `outputs/processed/outcome_features.csv.gz` | Số phiếu theo lớp, bình luận đã gộp và đặc trưng lịch sử đồ thị. |
| `outputs/processed/outcome_targets.csv` | Nhãn `-1/+1` và tập chia theo `election_id`. |
| `outputs/processed/stance_comments.csv.gz` | Bình luận đã làm sạch, nhãn phiếu `-1/0/+1` và tập chia. |
| `outputs/processed/signed_edges.csv.gz` | Sự kiện có dấu `-1/+1` theo thời gian để dựng đồ thị. |
| `outputs/tfidf/{outcome,stance}/` | Vectorizer `.joblib`, ma trận thưa `.npz` và bảng ID theo thứ tự hàng cho train, validation và test. |

Với outcome, nối đặc trưng và nhãn qua `election_id`. Dùng các cột `vote_*`, `graph_*` và ma trận TF-IDF ở `outputs/tfidf/outcome/` tùy mô hình so sánh. Với stance, `record_id` nối hàng TF-IDF với nhãn bình luận. Mỗi vectorizer được fit trên train của tác vụ tương ứng và áp dụng cho validation và test.

## Quy tắc chọn mẫu

- Tách tệp SNAP thành các cụm liên tiếp theo ứng viên. Dùng các thời điểm trên hồ sơ Wikipedia để ghép một-một với mã trang. So sánh `RES` với kết quả lưu trữ sau bước ghép.
- Tại mốc K, chọn các cụm có ngày đọc được cho mọi phiếu, có đủ K phiếu đầu khớp thời điểm trên hồ sơ và tránh trường hợp phiếu K và K+1 cùng phút. Mốc mặc định để xuất đặc trưng là 10.
- Chia theo thời điểm phiếu đầu quan sát được. Train trước 2008, validation trong 2008–2009, test từ 2010. Dùng giờ đóng đọc được hoặc ngày lưu trữ cộng 7 ngày để kiểm tra nhãn của tập trước đã có trước khi tập sau bắt đầu.
- Làm sạch wiki markup, chữ ký, đoạn gạch bỏ, nhãn phiếu đầu câu và các cụm từ tiết lộ trực tiếp kết quả. Tính đặc trưng đồ thị từ các cạnh xuất hiện trước phiếu đầu của lần ứng cử đang xét.

SNAP lưu các phiếu quan sát được và có thiếu thời gian, nhất là ở những năm đầu. Chữ ký trên Wikipedia đối chiếu được mốc phút của phiếu. Bình luận lưu trữ có thể chứa nội dung được sửa sau thời điểm chữ ký ghi nhận.

## Kết quả tái tạo

Từ 198.275 phiếu, pipeline ghép được 3.574 lần ứng cử. Có 1.066 mẫu hợp lệ tại mốc 10, 809 tại mốc 20 và 687 tại mốc 30. Có 672 mẫu đủ điều kiện ở cả ba mốc.

| Tập ở mốc 10 | Lần ứng cử | Thành công | Thất bại |
|---|---:|---:|---:|
| Train | 561 | 283 | 278 |
| Validation | 290 | 141 | 149 |
| Test | 215 | 87 | 128 |
