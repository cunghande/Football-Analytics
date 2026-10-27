# Kiểm tra bản xuất `detection-player` v4

Nguồn: `https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4`
File gốc: `D:\Downloads\detection player.v4-v.yolov8.zip`
SHA-256 của ZIP: `49d5bede1cc8835b8fef40c617682c6fc08903e753565756da7073bb2e5ffd4c`
Thư mục giải nén: `datasets/football_4class` (được `.gitignore` bỏ qua)
Giấy phép ghi trong `data.yaml` và README của ZIP: CC BY 4.0.

README của bản xuất ghi không áp dụng augmentation. `data.yaml` khai báo đúng bốn lớp theo thứ tự ID: `ball`, `goalkeeper`, `player`, `referee`.

| Tập | Ảnh | Box ball | Box goalkeeper | Box player | Box referee |
|---|---:|---:|---:|---:|---:|
| train | 1.844 | 1.478 | 1.352 | 18.426 | 2.206 |
| valid | 310 | 246 | 229 | 2.913 | 351 |
| test | 175 | 140 | 125 | 1.798 | 225 |

Số box ở bảng là số **hợp lệ theo `scripts/audit_dataset.py`**; các dòng lỗi chưa được tính. Tổng cộng có 2.329 file ảnh, nhưng chỉ có 2.090 nội dung ảnh khác nhau theo SHA-256. Công cụ phát hiện 53 dòng nhãn lỗi: 15 tọa độ/kích thước không hợp lệ, 28 box vượt biên ngang, 10 box vượt biên dọc. Có 75 nội dung ảnh xuất hiện ở hơn một tập; đây là số hash trùng xuyên tập, không phải số trận hoặc số frame gần giống.

Tên file cho thấy bảy nhóm clip (`1`, `2`, `3-_-2`, `bolhi`, `courtoi_1`, `pelanty`, `Video-sans-titre-Realisee-avec-Clipchamp-1-`) đều hiện diện trong cả train, valid và test. Không thể khẳng định các nhóm này là những trận độc lập chỉ dựa vào tên file. Vì vậy **không dùng split valid/test của bản xuất nguyên trạng để tuyên bố khả năng tổng quát hóa hoặc so sánh ba thiết lập ở bước B**.

Để có kết quả đánh giá chính thức: giữ nguyên bản ZIP và thư mục giải nén này; xác định clip nào thuộc cùng trận; tạo bản dữ liệu dẫn xuất với cách chia theo trận; sửa hoặc loại các box lỗi sau khi kiểm tra ảnh; chạy lại `python scripts/audit_dataset.py <thu_muc_du_lieu_moi>`. Nếu không xác định được trận nguồn, cần một video/trận khác, được gán nhãn, làm tập đánh giá độc lập và ghi rõ giới hạn của tập công khai trong báo cáo.

## Bản dẫn xuất tạm thời để fine-tune

Khi ID trận không có sẵn, `python scripts/prepare_clip_split.py` tạo `datasets/football_4class_clip_split` mà không sửa bản xuất gốc. Quy tắc chia theo tên clip:

- train: `Video-sans-titre-Realisee-avec-Clipchamp-1-`, `3-_-2`, `bolhi`, `courtoi_1`;
- valid: `1`, `2`;
- test: `pelanty`.

Script bỏ 239 bản sao ảnh giống hệt theo SHA-256, loại 15 box có diện tích bằng 0 và cắt 63 box vượt biên rất nhỏ về `[0, 1]` sau khi khử ảnh trùng. Bản dẫn xuất có 1.247 ảnh train, 599 ảnh valid và 244 ảnh test; lệnh audit chạy thành công, không còn ảnh trùng hệt giữa các tập hay tọa độ nhãn sai. `manifest.csv` ghi đường dẫn ảnh nguồn, hash và tập mới; `provenance.json` ghi phép chia và số chỉnh sửa.

Đây là **chia theo clip, không phải chia theo trận đã xác minh**. Tên clip `1` và `2` được giữ cùng một tập vì có thể liên quan; các tên khác cũng có thể thuộc cùng một trận. Các chỉ số validation/test từ bản này chỉ dùng để theo dõi thí nghiệm ban đầu. Để đáp ứng yêu cầu đánh giá độc lập theo trận, cần bổ sung nguồn video có ID trận rõ ràng hoặc tập gán nhãn từ trận khác.
