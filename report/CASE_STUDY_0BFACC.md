# Kiểm tra báo cáo trên video `0bfacc_0.mp4`

## Thiết lập

- Video mẫu được liệt kê trong [`examples/soccer/setup.sh` của Roboflow Sports](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh). Chỉ phân tích đoạn 0–18 giây: 450 frame, 25 FPS, thu từ 1920×1080 về 1280×720.
- Checkpoint: `detection_player_freeze10_v1/weights/best.pt`; suy luận YOLO trên NVIDIA RTX 3050 tại máy thử.
- Hai đội là cụm màu áo ẩn danh. Không có hiệu chỉnh sân, danh tính cầu thủ hoặc nhãn chuẩn vị trí bóng/quyền kiểm soát bóng cho clip này.

## Vấn đề đã quan sát và cách sửa

| Chỉ số từ pipeline | Trước sửa | Sau sửa | Cách hiểu |
|---|---:|---:|---|
| Frame phát hiện trực tiếp bóng | 0/450 | 93/450 (20,7%) | Mức quan sát của detector, **không** phải precision/recall |
| Frame gán bóng cho một đội | 0 | 73 | Theo khoảng cách bóng–cầu thủ, chưa được đối chiếu nhãn chuẩn |
| Frame không gán bóng | 450 | 377 | Không đồng nghĩa một đội khác kiểm soát bóng |
| Số track lớp player | 52 | 52 | Không phải 52 người: cùng người có thể nhận nhiều ID; người ngoài sân có thể bị phát hiện |

Nguyên nhân chính của số 0 bóng là ngưỡng chung 0,5 kết hợp với điều kiện tạo track mới của ByteTrack. Bóng nhỏ có nhiều dự đoán dưới ngưỡng này. Bản sửa dùng ngưỡng 0,25 **chỉ cho lớp ball**, giữ 0,5 cho các lớp người, rồi lấy dự đoán bóng có độ tin cậy cao nhất theo frame. Bóng nội suy chỉ để hiển thị; các frame nội suy không được tính là phát hiện trực tiếp hoặc dùng để gán bóng. Trong một bản kiểm tra trung gian, việc đếm cả nội suy làm tỷ lệ bóng thành 47,6%; số đúng cho phát hiện trực tiếp là 20,7%.

PDF mới ghi rõ model, đoạn thời gian, tỷ lệ phát hiện trực tiếp và số frame không gán bóng. Khi không có frame gán được bóng, bảng ghi **Không đo được** thay vì diễn giải 0%/100% là tỷ lệ kiểm soát. Bảng PDF chỉ hiện track quan sát ít nhất 1 giây; cả 52 track vẫn có trong CSV. Trong clip này có 16 track dưới 1 giây; một track được dự đoán là thủ môn chỉ xuất hiện 2 frame nên chưa thể xác nhận vai trò. Báo cáo không hiển thị quãng đường và tốc độ vì chưa hiệu chỉnh sân.

## Giới hạn thực nghiệm

Các con số trên là **mức quan sát và mức gán của pipeline**, không phải độ chính xác phát hiện bóng hay tỷ lệ kiểm soát bóng thực. Hạ ngưỡng có thể tăng phát hiện nhầm; cần gán nhãn thủ công một tập frame từ trận khác để đo precision, recall và lỗi gán bóng trước khi dùng các chỉ số này cho quyết định huấn luyện. Video góc truyền hình còn có người ngoài sân; chưa có lọc người trong ranh giới sân. Minimap hiện dùng tọa độ ảnh, không phải tọa độ mét thực.
