# Báo cáo thực nghiệm bước B — so sánh ba thiết lập YOLO26x

Đây là phần kết quả **đã chạy** trên Modal. [Bước A](BAO_CAO_BUOC_A.md) mô tả chuẩn bị dữ liệu và run fine-tune đầu tiên. Các file nguồn của bảng và hình nằm trong [`report/data/`](data/) và có thể kiểm tra độc lập với checkpoint lớn.

## 1. Câu hỏi và giả thuyết

Mục tiêu là so sánh bộ phát hiện pretrained gốc với hai cách chuyển học trên cùng dataset bóng đá bốn lớp. Giả thuyết ban đầu: fine-tune giúp phát hiện bóng đá tốt hơn pretrained COCO; đóng băng mười lớp đầu có thể phù hợp khi dữ liệu huấn luyện ít và mất cân bằng. Giả thuyết chỉ được đánh giá trên split hiện có, không đồng nghĩa với kết luận cho trận mới.

| Thiết lập | Huấn luyện trên dữ liệu bóng đá | Cách dùng |
|---|---|---|
| Pretrained gốc `yolo26x.pt` | Không | Ánh xạ `person → player`, `sports ball → ball` |
| `freeze=10` | Có | Khởi tạo từ cùng pretrained checkpoint; đóng băng mười lớp đầu |
| Fine-tune toàn bộ | Có | Khởi tạo từ cùng pretrained checkpoint; không đóng băng |

Ánh xạ baseline theo cách [mã gốc phân loại detections](../src/yolo_detector.py): `person` thuộc nhóm cầu thủ, `sports ball` thuộc nhóm bóng. COCO pretrained không có đầu ra riêng cho `goalkeeper` và `referee`, nên AP của hai lớp này là 0 do **khác hệ nhãn**, không nên diễn giải là nó đã học nhưng phát hiện kém. Điều này làm so sánh mAP bốn lớp với baseline có giới hạn cấu trúc; so sánh hai run fine-tune trực tiếp hơn. [Tài liệu YOLO26](https://docs.ultralytics.com/models/yolo26/) ghi pretrained detection dùng 80 lớp COCO; tên và ID `person`/`sports ball` được kiểm tra trực tiếp từ checkpoint khi đánh giá.

## 2. Dữ liệu và cấu hình

Nguồn dữ liệu là [detection-player v4](https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4), bản dẫn xuất đã loại ảnh trùng hệt và sửa tọa độ nhãn như [báo cáo bước A](BAO_CAO_BUOC_A.md) mô tả. Có 1.247 ảnh train, 599 ảnh validation và 244 ảnh test. Validation có 9.493 box; test có 998 box. Chia theo nhóm tên clip, **chưa xác minh độc lập theo trận**. Tập test chỉ chứa clip `pelanty`, góc quay phạt đền; vì vậy dùng để phân tích thay đổi miền dữ liệu, không xem là đại diện mọi cảnh bóng đá.

Hai run train đều dùng `yolo26x.pt`, `imgsz=640`, batch 8, GPU H100 trên Modal, seed 0, yêu cầu tối đa 50 epoch và early stopping `patience=20`. `freeze=10` khóa mười lớp đầu theo [tham số train của Ultralytics](https://docs.ultralytics.com/modes/train/); run này hoàn thành 33 epoch, chọn checkpoint epoch 13. Run fine-tune toàn bộ hoàn thành 43 epoch, chọn checkpoint epoch 23. Cả hai chỉ chạy **một seed**; không có độ lệch chuẩn hay khoảng tin cậy qua nhiều lần train.

Sau train, cùng một script [`scripts/evaluate_detectors.py`](../scripts/evaluate_detectors.py) chạy cả ba checkpoint trên từng split. Để tránh khác biệt do chế độ đệm ảnh mặc định, inference dùng `imgsz=640`, `rect=False`, chia batch thực tế thành nhóm 8 ảnh, confidence tối thiểu 0,001, tham số IoU 0,7, tối đa 300 dự đoán/ảnh. Box YOLO được đổi sang định dạng COCO, rồi tính AP bằng [COCOeval](https://github.com/cocodataset/cocoapi) với tối đa 100 dự đoán/ảnh, IoU 0,50–0,95 cách nhau 0,05. Precision/recall trong bảng là **trung bình bốn lớp** tại confidence ≥ 0,25 và ghép box tham lam ở IoU ≥ 0,50; chúng khác cách chọn điểm precision/recall trong log Ultralytics. Giao thức được lưu nguyên trong [`comparison_b_summary.json`](data/comparison_b_summary.json).

Số ở bảng B được tính lại từ cùng giao thức COCOeval, nên có thể lệch nhẹ so với dòng `results.csv` ở [bước A](BAO_CAO_BUOC_A.md). Đường cong train ở Hình 3 lấy từ log trong quá trình train, dùng để xem xu hướng, **không dùng** để điền bảng so sánh chính.

## 3. Kết quả chính trên validation chia theo clip

| Model | Precision* | Recall* | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 ball | AP@0.5:0.95 ball |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained COCO, ánh xạ hai lớp | 0,22781 | 0,24673 | 0,18948 | 0,06199 | 0,11348 | 0,03575 |
| `freeze=10` | **0,59359** | **0,57800** | **0,55910** | **0,27091** | **0,44256** | **0,16260** |
| Fine-tune toàn bộ | 0,58154 | 0,56304 | 0,52573 | 0,24734 | 0,40077 | 0,14078 |

\* Macro precision/recall tại confidence ≥ 0,25, IoU ≥ 0,50, không phải precision/recall do Ultralytics ghi ở mỗi epoch.

![Hình B1. So sánh chỉ số chính và AP bóng](figures/comparison_b_metrics.png)

*Hình B1.* Bốn thước đo trên cùng 599 ảnh validation, cùng phép đánh giá. Hai lớp chỉ có ở bộ dữ liệu bóng đá khiến baseline COCO bị giới hạn trong mAP bốn lớp. Bản in: [PDF](figures/comparison_b_metrics.pdf).

`freeze=10` cao hơn fine-tune toàn bộ 0,02357 điểm mAP@0.5:0.95 và 0,04179 điểm AP@0.5 của bóng trên split validation này. Vì chỉ có một seed, không được gọi chênh lệch đó là khác biệt có ý nghĩa thống kê. Kết quả phù hợp giả thuyết đóng băng có thể có lợi ở dữ liệu hiện tại, nhưng chưa chỉ ra cơ chế nguyên nhân.

| Model | AP@0.5 ball | AP@0.5 goalkeeper | AP@0.5 player | AP@0.5 referee |
|---|---:|---:|---:|---:|
| Pretrained COCO | 0,11348 | 0,00000 | 0,64443 | 0,00000 |
| `freeze=10` | 0,44256 | 0,17749 | 0,86394 | 0,75241 |
| Fine-tune toàn bộ | 0,40077 | **0,18806** | 0,81839 | 0,69570 |

![Hình B2. AP theo lớp](figures/comparison_b_per_class.png)

*Hình B2.* AP@0.5 của từng lớp trên validation. Goalkeeper còn thấp ở cả hai run fine-tune; full fine-tune nhỉnh hơn ở riêng lớp này, nên không nên tuyên bố `freeze=10` thắng ở mọi lớp. Bản in: [PDF](figures/comparison_b_per_class.pdf).

![Hình B3. Đường cong hai run train](figures/comparison_b_training_curves.png)

*Hình B3.* mAP validation do Ultralytics ghi theo epoch; dấu tròn là checkpoint tốt nhất của mỗi run. Đường cong dao động và hai run dừng ở số epoch khác nhau do early stopping. Hình này bổ trợ phân tích diễn biến học, còn Bảng 1 dùng lượt COCOeval chung. Bản in: [PDF](figures/comparison_b_training_curves.pdf).

## 4. Kết quả thăm dò trên clip test `pelanty`

| Model | Precision* | Recall* | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 ball |
|---|---:|---:|---:|---:|---:|
| Pretrained COCO | 0,04244 | 0,15502 | 0,11278 | 0,03617 | 0,01229 |
| `freeze=10` | 0,32586 | 0,44117 | 0,29635 | 0,10937 | 0,89423 |
| Fine-tune toàn bộ | 0,36239 | 0,31622 | 0,29436 | 0,11512 | 0,35558 |

\* Cùng định nghĩa như bảng validation. Tập này có 240 box ball, 244 goalkeeper, 244 player và 270 referee trên 244 ảnh; phân bố khác mạnh với validation. AP bóng của `freeze=10` cao nhưng AP goalkeeper/referee trên test gần 0, nên mAP tổng không cao tương ứng. Fine-tune toàn bộ nhỉnh hơn 0,00575 điểm mAP@0.5:0.95 trên clip test này, dù kém hơn trên validation. Sự đổi thứ hạng là dấu hiệu cần đánh giá thêm trên nhiều trận/góc quay; chưa đủ cơ sở để chọn model triển khai dựa vào tập test hẹp này. Không dùng tập test để chỉnh tham số hoặc chọn checkpoint rồi lại báo cáo nó như holdout độc lập.

## 5. Giới hạn và kết luận học thuật

1. Chỉ mới tách theo tên clip; ảnh giữa các split không trùng hash nhưng vẫn có thể cùng trận. Chưa đáp ứng tiêu chí đánh giá độc lập theo trận.
2. Pretrained COCO và mô hình bốn lớp khác không gian nhãn. So sánh bốn lớp cho thấy giá trị của việc học nhãn đặc thù, nhưng không phải thí nghiệm thuần túy chỉ thay đổi tham số freeze.
3. Mỗi cấu hình chỉ có một run; early stopping chọn epoch khác nhau. Cần lặp nhiều seed và đánh giá trên trận độc lập để ước lượng độ biến thiên.
4. Chất lượng nhãn bóng, người xem trong nền và nhãn bỏ sót có thể ảnh hưởng AP. Cần audit thủ công một mẫu lỗi và các frame test trước khi rút kết luận mạnh.
5. [Bước C](BAO_CAO_BUOC_C.md) đã đo chia đội với box gán nhãn, nhưng chưa đo ảnh hưởng end-to-end của detector mới lên tracking, chia đội, kiểm soát bóng và quãng đường.

**Kết luận ở phạm vi dữ liệu hiện tại:** hai model đã fine-tune đều đạt mAP/AP bóng cao hơn baseline COCO sau ánh xạ. Trên validation chia theo clip, `freeze=10` dẫn đầu mAP và AP bóng, còn full fine-tune dẫn nhẹ ở AP goalkeeper. Kết luận về khả năng tổng quát sang trận mới vẫn để mở.

## 6. Tái tạo và nguồn

```powershell
# Tập dữ liệu đã được upload lên Volume football-dataset.
python -X utf8 -m modal run modal_train.py --dataset football_4class_clip_split --model yolo26x.pt --epochs 50 --imgsz 640 --batch 8 --freeze 10 --name detection_player_freeze10_v2
python -X utf8 -m modal run modal_evaluate.py --freeze detection_player_freeze10_v2 --output evaluation_b_fixed640_v3
python -X utf8 scripts/make_comparison_figures.py
```

Lệnh đánh giá ở trên cần checkpoint full fine-tune đã có trên Volume. Nếu tái tạo từ đầu, chạy lệnh train full trong [báo cáo A](BAO_CAO_BUOC_A.md) với tên run phù hợp, rồi truyền `--full <ten_run>` cho `modal_evaluate.py`. Các hình hiện tại được tái tạo trực tiếp từ [`comparison_b_summary.json`](data/comparison_b_summary.json), [`freeze10_results.csv`](data/freeze10_results.csv) và [`full_results.csv`](data/full_results.csv); không cần kết nối Modal để vẽ lại. Ba checkpoint SHA-256 lần lượt là: baseline `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`, freeze `7f06b71d15e8914ee0281479948769606c7047df3bf00f5cef8d810563627f16`, full `77e9593c2a758ed31bedb2a3b993157fa4f1c01f9044f844d1ee798151501417`.

Nguồn tham khảo cho giao thức: [Ultralytics: train/freeze](https://docs.ultralytics.com/modes/train/), [Ultralytics: predict và `rect`](https://docs.ultralytics.com/modes/predict/), [Ultralytics: validation](https://docs.ultralytics.com/modes/val/), [COCO API và COCOeval](https://github.com/cocodataset/cocoapi). Nguồn code/dataset và giấy phép được ghi ở [báo cáo A](BAO_CAO_BUOC_A.md).
