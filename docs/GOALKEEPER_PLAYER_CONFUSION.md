# Sửa lỗi nhầm `player` và `goalkeeper`

## Kết quả kiểm tra video mới

Lần chạy `Screen Recording 2026-10-27 215224.mp4` phân tích 900 frame. Model dùng checkpoint `detection_player_freeze10_v1`. Dữ liệu YOLO/ByteTrack thô có 8.717 quan sát `player`, 125 quan sát `goalkeeper` và 639 quan sát `referee`; **15 track ID** từng nhận cả nhãn `player` lẫn `goalkeeper`. Đây là số quan sát và số ID, không phải số người hay số lỗi đã được chấm nhãn chuẩn.

Xem khung hình thấy thủ môn áo cam có lúc không có box `goalkeeper`, trong khi cầu thủ áo trắng/xanh có lúc bị gán `goalkeeper`. Video còn có cảnh cận và cắt góc quay; một track ID có thể nhảy sang người khác qua cảnh cắt. Vì vậy bỏ phiếu nhãn trên toàn track sẽ có nguy cơ biến một người bị nhầm lâu thành thủ môn và bỏ sót thủ môn thật. Chưa có nhãn thủ công đủ để tính tỷ lệ nhầm trên video này.

Model đã được fine-tune **bốn lớp**, nhưng không có danh sách cố định các màu áo thủ môn. Nó học hình dạng/ngữ cảnh/màu xuất hiện trong ảnh train. Bộ train đã xử lý có **897 box thủ môn so với 12.967 box cầu thủ**; checkpoint `freeze=10` đạt AP@0.5 của thủ môn **0,17749** trên validation, thấp hơn nhiều so với lớp cầu thủ **0,86394**. [Báo cáo thực nghiệm B](../report/BAO_CAO_BUOC_B.md) còn ghi AP thủ môn gần 0 trên clip test `pelanty` khác phân bố. Do đó không nên coi model hiện tại đã tổng quát tốt sang màu áo/góc quay mới.

## Dữ liệu cần bổ sung

Đã xuất **60 ảnh cách đều** từ video mới vào `datasets/role_review_screen_recording/` và ZIP cùng tên. Ảnh hiện **chưa có nhãn**; `manifest.csv` lưu frame, thời điểm và video nguồn. Script dùng lại cho video khác:

```powershell
python scripts/export_role_annotation_frames.py "D:\duong-dan\video.mp4" "datasets\role_review_video_2" --count 60
```

Để cải thiện model chung, nên thu 100–300 frame từ **nhiều trận và nhiều màu áo thủ môn**, gồm cả góc rộng, góc cận, cảnh thay góc, cầu thủ đứng gần cầu môn và trọng tài có áo khác đội. Gán **đầy đủ bốn lớp** `ball`, `goalkeeper`, `player`, `referee` trên từng ảnh. Thủ môn vẫn là `goalkeeper` dù màu áo giống một đội; cầu thủ thường đứng trong vòng cấm vẫn là `player`. Không tạo file nhãn rỗng cho ảnh chưa gán. Kiểm tra các box thủ môn nhỏ/bị che và lỗi cầu thủ bị gán thủ môn.

Giữ mọi frame của cùng trận trong **một nhóm chia tập**; không chia ngẫu nhiên frame gần nhau giữa train và validation/test. Nếu dùng video hiện tại để bổ sung train, cần một video/trận khác hoàn toàn để đánh giá khả năng tổng quát. Sau khi gán nhãn, fine-tune lại trên Modal bằng cùng bốn tên lớp và so sánh AP riêng `goalkeeper`, precision/recall và bảng nhầm `player`↔`goalkeeper` trên tập độc lập. Chỉ thay checkpoint mặc định nếu kết quả mới tốt hơn trên dữ liệu không dùng để chọn tham số.

## Giới hạn của sửa nhãn sau YOLO

Làm mượt vai trò theo vài frame có thể giảm nhãn nhảy trên **cùng người**, nhưng không tạo được box thủ môn khi YOLO bỏ sót và không sửa được nhầm ổn định. Video cắt cảnh còn làm track ID đổi hoặc nối sai người. Vì vậy dự án giữ một luồng chạy chung và ưu tiên cải thiện dữ liệu/model, thay vì thêm chế độ theo màu áo cho riêng một clip.
