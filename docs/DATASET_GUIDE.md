# Bước 3 — Chuẩn bị dataset bóng đá 4 lớp

Tài liệu này chuẩn bị dữ liệu cho bài tập lớn trước khi huấn luyện. Chưa có kết quả thực nghiệm nào được giả định ở đây. Repo gốc hiện có hướng dẫn cho **một dataset khác** trong `README.md`; không dùng số ảnh, giấy phép hay điểm mAP của dataset đó để mô tả dataset mới.

## 1. Chọn nguồn và tải bản xuất YOLO

- Nguồn chính do bạn chọn: [detection-player](https://universe.roboflow.com/fsdm-olss1/detection-player).
- Chỉ bổ sung khi cần: [football-objects-detection](https://universe.roboflow.com/ifap/football-objects-detection). Trên trang Roboflow, kiểm tra lại tên các lớp của **phiên bản tải về**; lớp bóng ở nguồn này có thể mang tên `soccer ball` và cần chuẩn hóa thành `ball`.
- Thêm 100–300 frame tự gán nhãn từ video demo bằng Roboflow hoặc CVAT. Giữ bản gốc, nhãn và danh sách `video_id`/`match_id` của từng frame.

Trên trang từng dataset, mở **Versions/Dataset**, chọn một phiên bản cụ thể và ghi lại số phiên bản, URL, giấy phép, tác giả, ngày truy cập, số ảnh gốc, số ảnh sau augmentation và danh sách lớp. Các thông tin này phải lấy từ trang của **đúng phiên bản**; không suy ra từ tên dự án. Trang Roboflow có mục **Download Dataset → Show download code** để đối chiếu cú pháp nếu giao diện thay đổi. [Hướng dẫn tải của Roboflow](https://docs.roboflow.com/roboflow/roboflow-jp/universe/universe-dtasettowodaunrdosuru).

Đầu vào: tài khoản Roboflow có quyền tải, API key của chính bạn và số phiên bản đã kiểm tra. Đầu ra: thư mục dataset YOLO chứa `data.yaml`, ảnh và nhãn. Có thể thất bại nếu phiên bản không tồn tại, không có quyền tải hoặc dự án không cho phép xuất định dạng đã chọn.

```powershell
python -m pip install roboflow
$datasetVersion = Read-Host "Nhap so phien ban da kiem tra"
python scripts/download_dataset.py --version $datasetVersion
```

Lệnh trên tải nguồn chính vào `datasets/football_4class` và lưu `source.json` (URL, phiên bản, định dạng; không chứa API key). API key được nhập kín lúc chạy; **không ghi key vào file hoặc Git**. Thư mục đích phải chưa tồn tại để tránh vô tình dùng lại bản tải cũ. Nếu không chắc số phiên bản, mở mục Versions của dataset trước khi chạy. Mã tương đương để dùng trong notebook:

```python
import getpass
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "fsdm-olss1"
PROJECT = "detection-player"
# Nếu tải nguồn bổ sung, đổi thành "ifap" và "football-objects-detection".
VERSION = int(input("Số phiên bản dataset đã kiểm tra trên Roboflow: "))
api_key = getpass.getpass("Roboflow API key: ")

rf = Roboflow(api_key=api_key)
destination = Path("datasets/football_4class").resolve()
dataset = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
    "yolov8", location=str(destination)
)
print("Dataset đã tải:", Path(dataset.location).resolve())
```

`"yolov8"` là tên định dạng xuất YOLO được dùng trong ví dụ SDK của Roboflow; nó không bắt buộc mô hình huấn luyện phải là YOLOv8. Kiểm tra `data.yaml` và tệp nhãn ngay sau khi tải. [Ví dụ Python của Roboflow](https://roboflow.com/blog/how-to-deploy-a-yolov8-model-to-a-raspberry-pi), [tài liệu Supervision/Roboflow](https://supervision.roboflow.com/latest/how_to/process_datasets/).

**Nếu không muốn dùng API key:** mở trang phiên bản dataset trong trình duyệt, chọn **Download Dataset → Download dataset → Continue → định dạng YOLO → Download ZIP**. Giải nén sao cho `data.yaml` nằm trực tiếp trong `datasets/football_4class`, rồi chạy lệnh audit ở mục 3. Cách này không dùng script Python nên bạn không phải nhập API key vào script; Roboflow có thể yêu cầu bạn đăng nhập trên website để tải. Ghi lại số phiên bản và giấy phép như hướng dẫn ở trên. [Các bước tải ZIP trên Roboflow](https://docs.roboflow.com/roboflow/roboflow-jp/universe/universe-dtasettowodaunrdosuru).

## 2. Cấu trúc thư mục và `data.yaml`

Sau khi chốt cách chia dữ liệu, cấu trúc mong muốn là:

```text
datasets/football_4class/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Ví dụ `data.yaml` (thay `path` bằng đường dẫn tuyệt đối trên máy hoặc Colab của bạn):

```yaml
path: D:/Code/football-analytics/datasets/football_4class
train: train/images
val: valid/images
test: test/images
nc: 4
names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
```

**Thứ tự ID chỉ là ví dụ.** Mỗi dòng nhãn YOLO phải theo `class_id x_center y_center width height`, bốn tọa độ đã chuẩn hóa vào `[0, 1]`. File `labels/ten_anh.txt` phải tương ứng với `images/ten_anh.jpg` (hoặc đuôi ảnh thực tế). Nếu bản tải về dùng thứ tự lớp khác, đổi cả ID trong **mọi** file nhãn khi chuẩn hóa; chỉ sửa `names` trong YAML sẽ làm nhãn sai. Nếu gộp nguồn bổ sung, thống nhất về đúng bốn tên `ball`, `goalkeeper`, `player`, `referee` và rà soát nhãn `soccer ball` → `ball`. [Định dạng detection của Ultralytics](https://docs.ultralytics.com/datasets/detect/).

Đầu vào: các ảnh và nhãn đã chốt. Đầu ra: một `data.yaml` và ba tập có cặp ảnh/nhãn tương ứng. Có thể sai khi đường dẫn không tồn tại, ID lớp không khớp `names`, tọa độ vượt `[0, 1]` hoặc ảnh thiếu nhãn.

## 3. Checklist kiểm tra trước khi train

Chạy công cụ kiểm tra cấu trúc và thống kê:

```powershell
python -m pip install PyYAML
python scripts/audit_dataset.py datasets/football_4class
```

Nó in số ảnh xuất ra và số box theo lớp cho từng tập; báo nhãn lỗi, ảnh thiếu file nhãn, và ảnh **trùng hệt byte** giữa các tập. Không dùng số ảnh xuất ra để khẳng định số ảnh gốc: ảnh do Roboflow augment có thể đã được lưu thành ảnh riêng. Để ghi **số ảnh thật**, lấy thống kê ảnh gốc/augmentation từ trang phiên bản Roboflow hoặc nhật ký nguồn trước khi tạo phiên bản. Ảnh đã augment chỉ được nằm trong train. [Hướng dẫn chia tập và augmentation của Ultralytics](https://docs.ultralytics.com/guides/preprocessing-annotated-data/).

- [ ] Ghi số ảnh gốc và số ảnh sau augmentation **riêng** cho train/valid/test; ghi rõ cách đếm.
- [ ] Xem số box và số ảnh có box cho từng lớp ở mỗi tập, đặc biệt `ball`, `goalkeeper`, `referee`; không chỉ nhìn tổng số ảnh.
- [ ] Xem thủ công các ảnh có bóng nhỏ, nhòe, bị che, gần đường biên; kiểm tra box ôm đúng bóng và bóng nhìn thấy nhưng bị bỏ nhãn.
- [ ] Kiểm tra lớp `player` không lẫn thủ môn/trọng tài; các nguồn phải có cùng quy ước gán nhãn.
- [ ] Kiểm tra ảnh thiếu nhãn, box ngoài biên, ID lớp sai, ảnh trùng và frame gần giống nhau giữa các tập. Hash trùng hệt **không phát hiện** mọi frame gần giống.
- [ ] So góc quay, độ phân giải, ánh sáng, kích thước bóng và kiểu video của dataset với video demo; lưu vài ảnh đối chiếu trong báo cáo.
- [ ] Ghi giấy phép của **từng** dataset và quyền sử dụng video demo. Giấy phép MIT của repo mã nguồn **không tự áp dụng** cho dữ liệu. Nếu không xác minh được giấy phép của một nguồn, chưa đưa nguồn đó vào bộ dữ liệu công bố.

## 4. Tạo validation/test độc lập theo trận

1. Lập bảng nguồn cho từng ảnh: `image_path, dataset, version, match_id, video_id, timestamp, original_or_augmented`. Nếu dataset công khai không có `match_id`, tìm thông tin nguồn hoặc gom thủ công theo video/trận; không tự nhận một phép chia là độc lập khi chưa biết nguồn ảnh.
2. Chia theo **match_id** trước khi tạo augmentation: mọi frame của cùng trận chỉ thuộc một trong `train`, `valid`, `test`. Các ảnh gần kề, ảnh cắt từ cùng frame và biến thể augmentation phải đi cùng tập gốc. Nếu bản xuất Roboflow đã có ảnh augment, truy lại ảnh gốc của từng biến thể và chia lại; nếu không truy được, không dùng tập valid đó để khẳng định tính độc lập.
3. Chỉ augment tập train. Giữ valid để chọn cấu hình; khóa test cho đánh giá cuối cùng. Kiểm tra mỗi tập có đủ bốn lớp, nhất là bóng.
4. Với 100–300 frame từ video demo: lấy frame cách nhau theo thời gian và gán nhãn đầy đủ. Nếu tất cả thuộc **một trận**, để toàn bộ trận đó làm test độc lập là phương án rõ ràng; không chia ngẫu nhiên frame của chính trận đó sang train/valid rồi gọi là đánh giá trên trận mới.
5. Nếu dữ liệu công khai không xác định được trận, ghi rõ giới hạn đó và dùng ít nhất một trận/video demo hoàn toàn tách biệt để đánh giá. Báo cáo riêng điểm trên valid công khai và trên video demo.

Đầu vào: ảnh gốc kèm nguồn trận/video. Đầu ra: danh sách train/valid/test cố định và có thể tái tạo. Có thể sai khi thiếu thông tin nguồn, nhiều dataset sao chép cùng ảnh, hoặc chia theo tên file mà thực chất các ảnh vẫn đến từ một trận. [Ultralytics về rò rỉ dữ liệu](https://docs.ultralytics.com/guides/model-testing/).

## 5. Ghi nguồn trong README và báo cáo

Tạo bảng nguồn tối thiểu như sau; điền từ trang dự án/phiên bản thực tế, **không điền ước lượng**:

| Nguồn | Tác giả/chủ sở hữu | URL dự án và phiên bản | Giấy phép | Ảnh gốc / ảnh augment | Vai trò |
|---|---|---|---|---|---|
| detection-player | kiểm tra trên Roboflow | URL + phiên bản đã tải | kiểm tra trên Roboflow | số liệu đã kiểm tra | chính |
| football-objects-detection (nếu dùng) | kiểm tra trên Roboflow | URL + phiên bản đã tải | kiểm tra trên Roboflow | số liệu đã kiểm tra | bổ sung |
| Frame video demo | người sở hữu video | tên video, trận, ngày trích frame | quyền sử dụng đã xác nhận | 100–300 frame gốc đã gán nhãn | kiểm tra độc lập hoặc train, theo cách chia đã chốt |

Trong báo cáo, trích dẫn **dataset** tách khỏi trích dẫn **repo gốc**. Roboflow thường có mục **Cite This Project** cung cấp BibTeX; sao chép đúng tác giả, tên, URL, năm từ trang tương ứng và thêm số phiên bản, ngày truy cập của bản bạn dùng. Ghi phần tự đóng góp: chọn/làm sạch dữ liệu, gán nhãn 100–300 frame, chuẩn hóa lớp và cách chia theo trận. Nếu trộn nguồn, lưu bảng số ảnh và box theo nguồn để kết quả có thể kiểm tra lại. [Hướng dẫn tải và giấy phép Roboflow Universe](https://docs.roboflow.com/roboflow/roboflow-jp/universe/universe-dtasettowodaunrdosuru).

**Điều kiện kết thúc bước 3:** tải được đúng phiên bản, lưu thông tin nguồn/giấy phép, có `data.yaml` 4 lớp khớp nhãn, thống kê ảnh gốc và box, và danh sách chia theo trận không rò rỉ. Sau đó mới bắt đầu bước A.
