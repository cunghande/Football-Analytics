# Bước A — fine-tune YOLO26x trên Modal

## Dữ liệu và cách chuẩn bị

- Nguồn: `detection-player` v4, giấy phép CC BY 4.0, bản xuất YOLOv8. ZIP gốc và SHA-256 được ghi trong [DATASET_AUDIT_V4.md](DATASET_AUDIT_V4.md).
- Bản train: `datasets/football_4class_clip_split`, tạo bằng `python scripts/prepare_clip_split.py` từ bản xuất gốc; giữ nguyên bốn lớp `ball`, `goalkeeper`, `player`, `referee`.
- 1.247 ảnh train, 599 ảnh valid, 244 ảnh test; không còn ảnh trùng hệt giữa các tập và không còn tọa độ nhãn sai theo `scripts/audit_dataset.py`.
- **Giới hạn:** chia theo tên clip, chưa xác minh độc lập theo trận. Mọi mAP bên dưới là chỉ số valid **tạm thời**, không phải kết quả tổng quát hóa theo trận.

## Cấu hình và kết quả

| Mục | Giá trị |
|---|---|
| Nơi chạy | Modal, H100 |
| Model nền | `yolo26x.pt` pretrained |
| Kích thước ảnh | 640 |
| Batch | 8 |
| Epoch yêu cầu | 50 |
| Early stopping | patience 20; dừng sau 43 epoch |
| Checkpoint | `runs/modal/detection_player_full_v1/weights/best.pt` |
| Precision trên valid | 0,65779 |
| Recall trên valid | 0,53901 |
| mAP@0.5 trên valid | 0,52506 |
| mAP@0.5:0.95 trên valid | 0,24651 |

Các số liệu ở bảng lấy từ dòng epoch 23 của `results.csv`, là dòng có mAP@0.5:0.95 cao nhất. Checkpoint `best.pt` đã nạp và chạy suy luận thành công trên một ảnh valid; model trả về đúng bốn tên lớp. File checkpoint có SHA-256 `77e9593c2a758ed31bedb2a3b993157fa4f1c01f9044f844d1ee798151501417`.

Thử nghiệm khởi động 3 epoch (`detection_player_smoke_v1`) cũng hoàn thành và lưu checkpoint, dùng để xác nhận đường chạy train. Bản 50 epoch mới là kết quả bước A. Chưa có AP riêng lớp ball hay điểm test đáng tin cậy để đưa vào bảng so sánh bước B; cần chạy cùng protocol cho model gốc, freeze=10 và fine-tune toàn bộ, tốt nhất trên tập đánh giá có trận nguồn rõ ràng.

## File và cách tái tạo

- Lệnh: `python -X utf8 -m modal run modal_train.py --dataset football_4class_clip_split --model yolo26x.pt --epochs 50 --imgsz 640 --batch 8 --name detection_player_full_v1`.
- Modal dataset Volume: `football-dataset:/football_4class_clip_split`.
- Modal results Volume: `football-training-runs:/detection_player_full_v1`.
- Bản local: `runs/modal/detection_player_full_v1/{args.yaml,results.csv,weights/best.pt}`. Thư mục `runs/` được `.gitignore` bỏ qua vì checkpoint lớn.
- Trang job Modal: `https://modal.com/apps/docung926/main/ap-zRrTKnmaSqwkpUOqFbdXs9` (cần đăng nhập tài khoản Modal tương ứng).

Đầu vào là bản dữ liệu dẫn xuất và pretrained checkpoint. Đầu ra là model fine-tune cùng log huấn luyện. Số liệu có thể sai lệch khi các clip thuộc cùng trận, khi nhãn nguồn thiếu/sai đối tượng, hoặc khi video ứng dụng có góc quay khác dataset. Bước B sẽ phải ghi rõ các giới hạn này thay vì coi split hiện tại là độc lập theo trận.
