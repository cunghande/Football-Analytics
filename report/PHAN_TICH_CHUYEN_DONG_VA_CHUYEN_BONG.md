# Kiểm tra chỉ số chuyển động và đường chuyền trên clip 750 frame

## Bối cảnh và nguyên nhân thiếu số liệu

PDF của `0bfacc_0.mp4` ghi 750 frame ở 25 FPS. Bóng được phát hiện trực tiếp trong 152/750 frame (20,3%); 611/750 frame không gán bóng cho đội. Quy tắc chuyền cũ đòi bóng nhìn thấy ở người chuyền, trong khoảng bóng tự do và ở người nhận, đồng thời cả hai người phải được gán bóng ổn định nhiều frame liên tiếp. Với bóng nhỏ và mất dấu ngắn, điều kiện này loại hết ứng viên. Đây là **thiếu bằng chứng theo quy tắc**, không phải bằng chứng trận không có đường chuyền.

Clip chưa có tọa độ sân hiệu chỉnh theo mét. Hệ số 0,1 mét/pixel cũ không phản ánh phối cảnh hoặc chuyển động camera; vì vậy không thể báo km/h hay quãng đường chạy thực từ clip này. Bản mới để trống các cột mét/km/h trong CSV và PDF khi chưa hiệu chỉnh sân.

## Cách tính mới

**Chuyển động tương đối trên ảnh.** Lấy vị trí chân của cùng track ở hai frame liên tiếp. Optical flow ước lượng độ tịnh tiến toàn cục của camera ở từng frame; cộng dồn độ dịch chuyển này rồi trừ khỏi tọa độ chân. Tổng độ dài các bước còn lại chia số bước và nhân FPS cho đơn vị pixel/giây. Không nối qua frame mất dấu. Ngưỡng bỏ qua chuyển động camera gần 0 giảm từ 2 xuống 0,15 pixel/frame; cache camera cũ được tách phiên bản để tính lại. Chỉ số này vẫn có sai số do phối cảnh, chuyển động camera không thuần tịnh tiến và rung box. Nó **không phải tốc độ thể lực** và không dùng để so sánh trực tiếp hai cầu thủ ở độ sâu ảnh khác nhau.

**Đường chuyền ứng viên.** Tìm hai lần gán bóng trực tiếp cho hai track khác nhau cùng đội, cách nhau tối đa 1,5 giây, có ít nhất một phía được quan sát từ hai frame gán bóng liên tiếp, cùng với ngưỡng khoảng cách ảnh. Bóng phải được thấy ở cả hai đầu; bóng có thể mất dấu ở giữa. CSV ghi `candidate_unverified_visible_flight` khi thấy bóng ở giữa, hoặc `candidate_unverified_sparse_ball` khi không thấy. Đây là **chuyển giao người được gán bóng theo quy tắc**, có thể là chuyền, tranh chấp, bóng bật hoặc lỗi track. Không có nhãn thủ công nên không gọi là chuyền thành công và không tính tỷ lệ chuyền chính xác.

## Kết quả tái chạy trên clip này

Chạy lại pipeline với detection stub đã lưu của chính clip (không chạy YOLO lần hai) cho **5 ứng viên**, gồm 2 trường hợp còn thấy bóng giữa hai đầu và 3 trường hợp bóng vắng ở giữa. Cả 5 thuộc nhóm áo `Đội 1` theo KMeans; các frame bắt đầu/kết thúc và track ID nằm trong `analysis_pass_candidates.csv`. Đây là đầu ra thuật toán, **không phải 5 đường chuyền đã xác nhận**. Trước sửa, CSV có 0 ứng viên do điều kiện quá chặt. Cần gán nhãn thủ công các pha này và cả những pha bị bỏ sót để đánh giá precision/recall.

Sau khi tính lại optical flow, 472/750 frame có dịch chuyển camera vượt ngưỡng mới; cache cũ ghi 0/750. CSV mới có `mean_camera_relative_motion_px_s` cho 38 track có đủ hai frame liên tiếp; ví dụ P0 là 39,1 px/s trong đoạn quan sát. Các cột `distance_m_estimate`, `mean_speed_kmh_estimate`, `max_speed_kmh_estimate` để trống vì clip chưa hiệu chỉnh sân. Không diễn giải 39,1 px/s thành km/h.

## Điều kiện để đo km/h

Cần biết kích thước sân thật và ánh xạ vị trí chân trên từng frame sang tọa độ sân theo mét. Với camera cố định, có thể hiệu chỉnh thủ công ít nhất bốn điểm sân nhìn rõ, kiểm tra sai số tái chiếu và đối chiếu tốc độ với phép đo tham chiếu. Với camera quay lia hoặc đổi góc, một phép hiệu chỉnh cố định cho cả clip không đủ; cần theo dõi biến đổi sân theo thời gian. Dự án hiện chưa tự hiệu chỉnh sân và không suy diễn km/h từ video chưa hiệu chỉnh.

Đầu ra kiểm tra cục bộ: `output/pdf/bao_cao_buoi_tap_chuyen_dong_duong_chuyen.pdf` và `output/dashboard_jobs/metrics_750_check_v2/analysis_report.zip`. ZIP có bảng CSV, pass map và biểu đồ chuyển động tương đối theo track quan sát ít nhất 5 giây. Biểu đồ chỉ mô tả pixel/giây, không xếp hạng thể lực. Các tệp video/PDF từ người dùng không được đưa lên GitHub.
