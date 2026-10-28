# Hướng dẫn chạy video và đọc báo cáo

Dashboard xử lý video trên máy đang mở ứng dụng. Modal chỉ dùng cho các lượt huấn luyện/đánh giá model. Trên máy hiện tại, checkpoint mặc định là `detection_player_role_ft_full_v1`, gồm bốn lớp `ball`, `goalkeeper`, `player`, `referee`.

## 1. Kiểm tra model và khởi động

Mở PowerShell tại `D:\Code\football-analytics`, sử dụng môi trường Python đã cài dependency, rồi chạy:

```powershell
python -m streamlit run dashboard.py
```

Giao diện tự chọn GPU 0 nếu PyTorch trong môi trường đang dùng nhận CUDA; nếu không, ứng dụng chạy CPU. Trên máy này, `venv` đã được cài bản PyTorch có CUDA. Dừng phiên Streamlit cũ bằng `Ctrl+C` rồi chạy lại khi cập nhật code hoặc thư viện.

Nếu thiếu checkpoint và đã có quyền truy cập Modal Volume:

```powershell
python -m pip install modal
python -X utf8 -m modal setup
New-Item -ItemType Directory -Force .\runs\modal\detection_player_role_ft_full_v1\weights | Out-Null
python -X utf8 -m modal volume get football-training-runs /detection_player_role_ft_full_v1/weights/best.pt .\runs\modal\detection_player_role_ft_full_v1\weights\best.pt --force
```

Checkpoint được loại khỏi Git do kích thước lớn. Không gửi API key hoặc thông tin đăng nhập vào chat hay commit.

## 2. Phân tích video

1. Mở địa chỉ Streamlit hiển thị trong PowerShell, thường là `http://127.0.0.1:8501`.
2. Kiểm tra tên model và dòng **Thiết bị suy luận**. GPU chỉ tăng tốc YOLO; các bước xử lý khác vẫn dùng CPU.
3. Chọn upload MP4/MOV/AVI/MKV tối đa 500 MB, hoặc chọn đường dẫn tới tệp nằm trên chính máy chạy Streamlit. Video cần đọc được và có FPS hợp lệ.
4. Chọn giây bắt đầu, bấm **Phân tích video** và chờ kết quả. Hệ thống tự chọn tối đa 900 frame liên tiếp và giảm độ phân giải theo ngân sách RAM. Thời lượng phụ thuộc FPS: 900 frame ở 25 FPS tương đương tối đa 36 giây.
5. Xem video kết quả và tải **Báo cáo buổi tập (PDF)** cùng **Báo cáo phân tích (ZIP)**. Báo cáo ghi đoạn thời gian, kích thước và FPS thực tế được xử lý; không suy diễn các con số cho cả video nguồn.

Tệp mỗi lượt nằm tại `output/dashboard_jobs/<mã-lượt>/`. PDF trong ZIP là kết quả của lần chạy tương ứng; PDF cũ không tự thay đổi khi cập nhật model.

## 3. Đọc kết quả

- `P`, `GK`, `R` là track dự đoán cầu thủ, thủ môn và trọng tài. Track ID không phải số áo hoặc danh tính; mất dấu/cắt cảnh có thể sinh ID mới hoặc ghép nhầm.
- Hai đội là cụm màu áo, không tự có tên đội. Thủ môn chưa được tự gán đội đáng tin cậy. Số track không phải số cầu thủ duy nhất.
- PDF ghi tỉ lệ frame có bóng được phát hiện trực tiếp và frame được quy tắc gán bóng cho đội. Nếu không gán được, hiển thị **Không đo được**. Bóng nội suy chỉ hỗ trợ hiển thị.
- Cột chuyển động trên ảnh là pixel/px/s, không phải thể lực hoặc tốc độ km/h. Khi chưa hiệu chỉnh sân, cột mét/km/h để trống. Camera lia, zoom và phối cảnh vẫn có thể gây sai số.
- Chuyền/nhận là **ứng viên** do quy tắc sinh ra, không phải đường chuyền thành công đã xác minh. Cần xem lại video và nhãn tham chiếu trước khi kết luận.
- Bộ lọc người ngoài sân và quy tắc ổn định vai trò tự hoạt động theo điều kiện của video, có thể bỏ sót hoặc sửa nhầm. Trong ZIP, xem `analysis_field_person_filter.json` và `analysis_role_refinement.json` để biết phương pháp và số quan sát thay đổi.

PDF chỉ hiển thị track quan sát ít nhất một giây; CSV giữ các track ngắn hơn. Tài liệu phân tích cách tính: [chuyển động và đường chuyền](../report/PHAN_TICH_CHUYEN_DONG_VA_CHUYEN_BONG.md).

## 4. Kiểm thử và sự cố thường gặp

```powershell
python -m pytest -q -m "not slow"
```

- **Thiếu checkpoint:** kiểm tra đúng đường dẫn và tải từ Volume ở bước 1.
- **Không thấy GPU:** kiểm tra PyTorch trong môi trường Python đang chạy dashboard đã được cài bản CUDA tương thích với GPU và driver. Khởi động lại Streamlit sau khi thay thư viện.
- **Không mở được video:** thử video có codec thông dụng và FPS cố định; kiểm tra đường dẫn nằm trên máy chạy dashboard.
- **Cổng 8501 đang dùng:** dừng phiên cũ hoặc dùng `python -m streamlit run dashboard.py --server.port 8502`.
- **Nhiều track hoặc nhầm vai trò:** đối chiếu video, vì track có thể đổi ID và detector/rule vẫn nhầm. Không suy ra số cầu thủ duy nhất từ tổng track.
- **Không có mét/km/h:** đây là trạng thái khi chưa hiệu chỉnh sân; px/s không thể đổi trực tiếp sang km/h bằng một hằng số chung.
- **Bóng ít hoặc nhiều frame chưa rõ:** xem trực tiếp video trước khi diễn giải gán bóng và các ứng viên chuyền.

Tránh chạy hai pipeline ghi cùng tệp trong `output/`. Dashboard tạo thư mục riêng cho từng lượt và không đọc cache tracker cũ. `python main.py` dùng đường dẫn mặc định trong `config.py` và có tùy chọn dùng `stubs/`; đổi video/model thì tắt `VIDEO_CONFIG["read_from_stub"]` để tránh tái sử dụng kết quả cũ.
