# Bước A — fine-tune YOLO trên Modal

> Hướng dẫn thí nghiệm A ban đầu. Checkpoint dashboard hiện tại được huấn luyện tiếp từ `freeze=10`; xem [thí nghiệm fine-tune vai trò](../report/BAO_CAO_FINE_TUNE_VAI_TRO.md) để biết dữ liệu, cấu hình và kết quả mới hơn.

Chỉ dùng dataset `detection-player` đã xuất sang YOLO và chuẩn hóa về đúng bốn lớp `ball`, `goalkeeper`, `player`, `referee`. Dataset `datasets/football` có sẵn trong repo thuộc nguồn khác và không được dùng thay cho thí nghiệm này. Bản `datasets/football_4class_clip_split` được tạo từ v4 bằng `scripts/prepare_clip_split.py`: loại ảnh trùng, sửa box vượt biên rất nhỏ và chia theo tên clip. **Không có ID trận nguồn, nên đây chỉ là split theo clip, chưa chứng minh độc lập theo trận.** Cách tải và kiểm tra giấy phép: [DATASET_GUIDE.md](DATASET_GUIDE.md).

## 1. Kiểm tra dataset ở máy

Đầu vào: thư mục `datasets/football_4class_clip_split` có `data.yaml`, `train/images`, `valid/images`, nhãn tương ứng và tập `test`. Đầu ra: báo cáo số ảnh/box theo lớp và cảnh báo nhãn. Có thể sai khi ID lớp không khớp YAML hoặc các clip thực chất cùng một trận.

```powershell
python scripts/audit_dataset.py datasets/football_4class_clip_split
```

Nếu script báo lỗi, sửa dataset trước khi tải lên Modal. Script không thể xác minh giấy phép, số ảnh gốc hoặc frame gần giống; đối chiếu các mục đó với metadata nguồn và checklist trong hướng dẫn dataset.

## 2. Tải dataset lên Modal Volume

Đầu vào: bản dẫn xuất đã kiểm tra và tài khoản Modal đang đăng nhập. Đầu ra: `/football_4class_clip_split/data.yaml` cùng ảnh/nhãn trong Volume `football-dataset`. Có thể sai khi tải thiếu thư mục hoặc dùng nhầm Modal profile.

```powershell
python -m modal profile current
python -m modal volume put football-dataset .\datasets\football_4class_clip_split /football_4class_clip_split
python -m modal volume ls football-dataset /football_4class_clip_split
```

`modal_train.py` kiểm tra bốn tên lớp và đường dẫn train/val trước khi khởi động Ultralytics. [Modal Volume CLI](https://modal.com/docs/cli/latest/volume).

## 3. Chạy thử, rồi fine-tune

Đầu vào: dataset trong Volume và checkpoint pretrained `yolo26x.pt`. Đầu ra: checkpoint và file kết quả trong `football-training-runs`. Có thể thất bại khi hết quyền GPU, lỗi cài thư viện, hết thời gian hoặc nhãn không hợp lệ. Modal đang cấu hình GPU H100; chạy GPU có thể phát sinh chi phí, xem trong tài khoản trước khi chạy. [Modal GPU](https://modal.com/docs/guide/gpu).

Chạy thử 3 epoch để xác nhận pipeline train và lưu được checkpoint:

```powershell
python -X utf8 -m modal run modal_train.py --dataset football_4class_clip_split --model yolo26x.pt --epochs 3 --imgsz 640 --batch 8 --name detection_player_smoke_v1
python -m modal volume ls football-training-runs /detection_player_smoke_v1
```

Sau khi chạy thử thành công, chạy fine-tune toàn bộ. Giữ nguyên dữ liệu, checkpoint nền, `imgsz` và batch khi làm các thí nghiệm so sánh ở bước B; tên run phải mới để không ghi đè kết quả:

```powershell
python -X utf8 -m modal run modal_train.py --dataset football_4class_clip_split --model yolo26x.pt --epochs 50 --imgsz 640 --batch 8 --name detection_player_full_v1
python -m modal volume ls football-training-runs /detection_player_full_v1
```

Đây là một cấu hình khởi đầu, chưa phải kết quả tối ưu. Nếu GPU báo thiếu bộ nhớ, giảm `--batch` và ghi lại thay đổi; bước B phải dùng cấu hình tương ứng để so sánh công bằng. Ultralytics lưu `weights/best.pt`, `args.yaml` và `results.csv` trong thư mục run khi huấn luyện thành công. [Ultralytics train mode](https://docs.ultralytics.com/modes/train/).

## 4. Tải kết quả về máy

```powershell
New-Item -ItemType Directory -Force .\runs\modal | Out-Null
python -X utf8 -m modal volume get football-training-runs /detection_player_full_v1 .\runs\modal
```

Kiểm tra `runs/modal/detection_player_full_v1/weights/best.pt`, `args.yaml` và `results.csv`. Nếu CLI đặt thư mục ở vị trí khác, dùng `modal volume ls` và kiểm tra nội dung thư mục tải về. Lưu URL và phiên bản dataset, số ảnh gốc từng tập, tham số train, thời gian chạy và checkpoint trong nhật ký thí nghiệm. **Chưa kết luận chất lượng model từ kết quả train; bước B sẽ đánh giá cùng một tập test cho ba thiết lập.**
