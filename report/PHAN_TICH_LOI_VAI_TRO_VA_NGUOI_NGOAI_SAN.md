# Kiểm tra lỗi vai trò và người ngoài sân trên clip 750 frame

## Dữ liệu và lỗi ban đầu

Clip phân tích có tên nguồn `0bfacc_0.mp4`, gồm 750 frame, 25 FPS và độ phân giải 990 × 556. Lần chạy cũ bật ô “Sửa nhãn trọng tài áo vàng/quần tối”, nhưng `analysis_role_refinement.json` ghi `raw_model_roles`: điều kiện an toàn của quy tắc cũ đã tắt việc sửa nhãn vì tỷ lệ mẫu áo vàng/quần tối là 302/3101 = 9,74%, cao hơn ngưỡng 8%. Vì vậy dấu chọn trong giao diện **không bảo đảm** quy tắc đã chạy; phải xem trường `method` trong JSON.

Xem trực quan các track 13, 37, 52 thấy người mặc áo vàng/quần tối bị thay đổi nhãn `P`/`R` theo frame. Track 8 của trợ lý trọng tài ở biên sân chủ yếu bị ghi `P`. Một số nhân viên áo phản quang, người ngồi khu kỹ thuật và người mặc đồ đen ngoài sân cũng bị ghi `P`. Đây là các trường hợp đối chiếu bằng hình; chưa có bộ nhãn chuẩn cho toàn clip.

## Phương pháp sửa

1. Sau ByteTrack, tạo mặt nạ sân từ thành phần liên thông xanh lớn nhất trong HSV. Với mỗi box người, đo khoảng cách từ điểm chân đến vùng sân. Quan sát ở xa sân hơn 8 pixel bị đề nghị loại. Người mặc áo tối ở sát mép dưới ảnh cũng được đề nghị loại. Bỏ phiếu các quan sát lân cận cùng track và loại cả track nếu đa số rõ ràng ở ngoài sân. Chỉ áp dụng cho người; bóng giữ nguyên. Tắt được bằng ô “Bỏ người ở khu kỹ thuật và ngoài sân khỏi thống kê (thử nghiệm)”.
2. Lấy mẫu màu áo/quần mỗi 5 frame. Khi bộ áo vàng/quần tối hơi phổ biến nhưng vẫn có dự đoán trọng tài hỗ trợ, dùng tỷ lệ trang phục trên **cùng track** để sửa nhãn. Track cần ít nhất 5 mẫu, ít nhất 3 mẫu khớp trang phục và ≥70% mẫu khớp, hoặc ít nhất 2 dự đoán trọng tài cùng ≥25% mẫu khớp. Sau đó bỏ phiếu tối đa 5 quan sát gần nhau. Các track không khớp trang phục nhưng model lẻ tẻ gán `R` được trả về `P` nếu có đủ mẫu. Kết quả và điều kiện kích hoạt lưu trong `analysis_role_refinement.json`.

Đây là bộ lọc ảnh và quy tắc trang phục, **không phải** mô hình phân đoạn sân hay bộ phân loại trọng tài đã huấn luyện. Không có hiệu chỉnh hình học sân.

## Kết quả chạy lại và cách diễn giải

Chạy lại pipeline trên 750 frame bằng **detection stub đã lưu từ chính clip này**, không suy luận YOLO lần hai. Bộ lọc loại 643 quan sát người và 26 track hoàn toàn; số track từng được gán `player` giảm từ 66 xuống 38; trung bình số đối tượng `player` mỗi frame giảm từ 20,7 xuống 18,3. Chế độ sửa nhãn ghi `supported_yellow_referee_tracks`, sau sửa có 5 track từng được gán trọng tài. Các số trên mô tả **đầu ra của thuật toán**, không phải precision/recall hay số người thực tế. Track ID có thể phân mảnh, nên 38 track không có nghĩa là 38 cầu thủ.

Kiểm tra hình trước/sau cho thấy trợ lý trọng tài track 8 và trọng tài track 37 chuyển từ `P` sang `R`, còn các nhân viên ở ngoài sân như track 26, 31, 44 và người khu kỹ thuật như track 57, 63 không còn tham gia thống kê. Cần kiểm tra tiếp bằng nhãn thủ công độc lập trước khi khẳng định cải thiện độ chính xác trên trận khác.

## Giới hạn và quy trình đánh giá tiếp

- Cầu thủ mặc áo tối ở sát biên dưới có thể bị loại nhầm; người ngoài sân đứng trên cỏ có thể còn sót. Màu sân/ánh sáng khác clip này có thể làm mặt nạ xanh thất bại.
- Đội mặc áo vàng, thủ môn mặc vàng/quần tối hoặc trọng tài mặc màu khác có thể làm quy tắc trang phục sai hoặc không kích hoạt.
- Nên chọn frame rải đều từ clip mới, đánh dấu thủ công các box theo `player`, `referee`, `goalkeeper`, `off_field_person`; đối chiếu trước/sau bằng precision, recall và bảng nhầm lẫn theo lớp. Giữ trận đánh giá độc lập với clip dùng chọn ngưỡng. Chỉ khi đó mới báo cáo được mức cải thiện định lượng.

Tệp kiểm tra trên máy: `output/pdf/bao_cao_buoi_tap_750_frame_da_loc.pdf`, `output/dashboard_jobs/field_role_750_check/analysis_browser.mp4` và `output/dashboard_jobs/field_role_750_check/analysis_report.zip`. Các tệp video/PDF kết quả được loại khỏi Git do kích thước và dữ liệu clip.
