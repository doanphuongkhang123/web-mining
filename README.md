# Wikipedia RfA toxicity · Web Mining

Pipeline Python xử lý phiếu và phân tích mức độ toxicity trong các lần ứng cử quản trị viên Wikipedia. Báo cáo gồm bốn biểu đồ cho giai đoạn 2004–2012.

## Chạy pipeline

Dùng Python 3.12. Đặt file điểm toxicity ở `data/wikipedia-rfa-toxicity.zip`, rồi chạy từ thư mục gốc của repo:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m votelens
```

Script tự tải và kiểm tra SHA-256 của file SNAP khi chưa có bản cục bộ.

| File đầu vào | Vai trò |
|---|---|
| `wiki-RfA.txt.gz` | Phiếu SNAP để nhóm lần ứng cử và làm sạch bình luận. |
| `wikipedia-rfa-toxicity.zip` | Bảng phiếu có cột `toxicity` để chạy EDA. |

Có thể chỉ định file ZIP hoặc CSV khác bằng `--scored`. Bảng điểm cần các cột `textid`, `tgt`, `yea`, `res`, `vote_start`, `src`, `vot`, `txt` và `toxicity`. Trong bảng này, `vote_start` là số ngày kể từ 01/01/1970, còn `toxicity` nằm trong khoảng 0–1.

Chạy riêng từng phần với file đã tải hoặc trích xuất:

```sh
python -m votelens.preprocess --data data/wiki-RfA.txt.gz
python -m votelens.eda --scored data/wikipedia-rfa-toxicity.zip
```

## Preprocessing

1. Đọc bảy trường SNAP thành bảng, gán `textid` theo vị trí bản ghi và tách `DAT` thành ngày, giờ.
2. Gom theo `tgt`, `yea` và `res`, rồi sắp phiếu theo ngày và giờ. Mở cụm mới khi hai phiếu liên tiếp trong nhóm cách nhau **hơn 7 ngày**. `vote_start` là ngày của phiếu mở cụm và được điền xuống trong cùng nhóm.
3. Dùng khóa `(tgt, yea, res, vote_start)` để đếm một lần ứng cử. Đây là khóa suy ra cho phân tích hồi cứu. `res` là kết quả cuối đã biết trong dữ liệu.
4. Dùng parser wikitext để giữ chữ trong đoạn in đậm, chữ nghiêng và nhãn hiển thị của liên kết. Bỏ URL, nội dung gạch ngang, nhãn phiếu đầu dòng và chữ ký khi nhận diện được. Giữ `txt` gốc trong bảng phiếu để đối chiếu qua `textid`.

`comments_for_scoring.csv.gz` lưu `txt_clean`, `text_status` và `text_flags` theo `textid`. Dùng trạng thái `ready` để chọn văn bản chấm điểm. Trạng thái `empty` và `symbols_only` đánh dấu dòng rỗng hoặc chỉ còn ký hiệu. Trạng thái `needs_review` giữ những trường hợp như template chưa nhận diện, chữ ký chưa rõ hoặc markup còn sót để kiểm tra. `summary.json` thống kê các trạng thái theo từng loại phiếu và ghi phiên bản cleaner.

Báo cáo EDA đọc riêng bảng điểm toxicity cục bộ. Khi chấm lại, cần tính điểm trên đúng phiên bản `txt_clean` được dùng.

## Kết quả

File chính là `outputs/rfa_toxicity/eda/report.html`. Mở file trong thư mục đầu ra để xem bảng minh họa và bốn biểu đồ.

Quy tắc 7 ngày tạo 4.126 khóa lần ứng cử suy ra từ SNAP. Trong giai đoạn 2004–2012, bảng điểm có 79.908 bản ghi, tương ứng 1.950 khóa lần ứng cử và 78.733 bình luận có điểm toxicity.

| Đầu ra trong `outputs/rfa_toxicity/` | Nội dung |
|---|---|
| `preprocessed/votes.csv.gz` | Phiếu SNAP sau khi định dạng lại và gắn `vote_start`. |
| `preprocessed/comments_for_scoring.csv.gz` | `textid`, bình luận sạch, trạng thái chấm và cờ cần kiểm tra. |
| `preprocessed/summary.json` | Số phiếu, khóa lần ứng cử, ngày thiếu và thống kê chất lượng text theo loại phiếu. |
| `eda/01_rfas_by_year.png` | Số lần ứng cử theo năm và kết quả. |
| `eda/02_support_share.png` | Tỷ lệ ủng hộ trong số phiếu ủng hộ và phản đối, kèm trung vị theo năm và kết quả. |
| `eda/03_toxicity_by_outcome.png` | Phân bố điểm toxicity theo kết quả ứng cử. |
| `eda/04_toxicity_by_vote.png` | Phân bố điểm toxicity theo loại phiếu. |
| `eda/*.csv`, `eda/summary.json` | Dữ liệu tổng hợp cho biểu đồ và thống kê số mẫu. |

Các năm 2003 và 2013 chỉ được ghi nhận một phần trong SNAP, nên EDA tập trung vào các năm 2004–2012.
