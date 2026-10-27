# Chạy thử báo cáo buổi tập

## Đầu ra hiện có

Sau khi phân tích một video, pipeline tạo `analysis_session_report.pdf` với thông tin video, số frame/FPS, mức độ phát hiện bóng, bảng theo track ID, tổng hợp theo hai nhóm áo và heatmap theo đội. Báo cáo **chỉ mô tả chỉ số**, không chấm điểm, xếp loại hay suy ra họ tên/số áo. PDF được tạo cả khi chạy `python main.py` và khi dùng dashboard. Dashboard còn cho tải PDF riêng và ZIP gồm PDF, CSV, hình, JSON phương pháp. Dữ liệu từ mỗi lần upload ở `output/dashboard_jobs/<id>/`, không được commit lên Git.

Khi chưa có bốn cặp điểm hiệu chỉnh sân trong `config.py`, PDF **không hiện mét và km/h**. Bảng theo track có thời gian nhìn thấy (giây), dịch chuyển vị trí chân trên ảnh (pixel), frame gán bóng, số đường chuyền ứng viên đi/đến. Pixel **không phải quãng đường chạy** vì camera di chuyển và phối cảnh. Đường chuyền ứng viên **chưa được xác minh thành công**; không có mẫu số tin cậy để tính tỷ lệ chuyền thành công. CSV vẫn có các cột ước tính mét/km/h từ hệ số cũ để kiểm tra công thức, nhưng không được đọc là số đo thực. Track ID có thể đổi khi mất dấu; số track không phải số người duy nhất. `possession_frames` là số frame được quy tắc gần bóng gán, không phải nhãn kiểm soát bóng đã xác minh.

## Chạy với video thật

1. Mở PowerShell tại thư mục repo. Cài dependency một lần:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Xác nhận checkpoint đã tải về từ Modal:

   ```powershell
   Test-Path .\runs\modal\detection_player_freeze10_v1\weights\best.pt
   ```

   Lệnh cần trả `True`. Nếu `False`, lấy `best.pt` của run `detection_player_freeze10_v1` từ Modal Volume `football-training-runs` và đặt đúng thư mục trên. Checkpoint lớn được loại khỏi GitHub.

3. Chạy dashboard:

   ```powershell
   python -m streamlit run dashboard.py
   ```

   Mở `http://127.0.0.1:8501`. Chọn checkpoint `Fine-tune freeze=10`. Trên máy đã nhận CUDA, chọn `GPU 0 — NVIDIA GeForce RTX 3050 ... (khuyến nghị)`; nếu mục này không hiện, thoát `venv` và khởi động dashboard bằng Python hệ thống. Chọn **Tải lên từ trình duyệt** cho MP4/MOV/AVI/MKV tối đa 500 MB, hoặc **Đường dẫn trên máy này** để đọc tệp lớn hơn mà không upload (ví dụ `D:\Downloads\video.mp4`, chỉ áp dụng khi Streamlit chạy trên cùng máy). Chọn tối đa **450 frame ở 720p** hoặc **900 frame với độ phân giải tự giảm**; có thể chọn giây bắt đầu. Video phải có FPS hợp lệ. Với video thật, **không chọn** ô “Clip ghép từ ảnh hoặc FPS giả định”. Bấm **Phân tích video**. PDF/CSV chỉ mô tả đoạn đã phân tích, không phải toàn bộ video nguồn. Sau khi đổi `.streamlit/config.toml`, dừng và khởi động lại Streamlit để áp dụng giới hạn upload mới.

4. Kiểm tra video kết quả, sau đó tải **Báo cáo buổi tập (PDF)** và **báo cáo phân tích (ZIP)**. Trong PDF, đối chiếu số frame/FPS với video. Mở CSV trong ZIP để kiểm tra các số gốc. Nếu tỷ lệ frame phát hiện bóng thấp hoặc frame “chưa rõ” cao, phần gán bóng cần xem lại trên video.

Sau khi cập nhật mã, hãy **chạy phân tích lại video** để có các cột mới `observed_seconds`, `observed_path_px`, `pass_candidate_sent`, `pass_candidate_received` và khung minimap nhỏ hơn. PDF cũ không tự cập nhật.

Nếu chạy không qua dashboard, đặt video tại `input/DFL-Scoutingfeed.mp4`, chọn `YOLO_CONFIG["model_path"]` trong `config.py`, rồi chạy `python main.py`. Các kết quả nằm trong `output/`, gồm `output/analysis_session_report.pdf`.

## Thử khi chưa có video demo

Máy đang có bản dataset cục bộ và checkpoint có thể chạy thử nhanh với **3 frame**:

```powershell
python scripts/smoke_session_report.py
```

Mở `output/dashboard_jobs/session_report_smoke/analysis_session_report.pdf`. PDF phải ghi **CLIP KIỂM TRA KỸ THUẬT**, 3 frame, FPS 10 *giả định* và không hiển thị tốc độ/quãng đường. File `analysis_report.zip` phải có PDF, CSV, JSON và hình. Nếu muốn chạy thêm một lượt 50 frame:

```powershell
python scripts/make_dataset_demo50.py
```

PDF của lượt 50 frame phải ghi 50 frame, FPS 25 *giả định*. Có thể upload `output/dashboard_jobs/session_report_smoke/input.mp4` hoặc `output/dashboard_jobs/dataset_demo50/input.mp4` vào dashboard để thử giao diện; **chọn ô “Clip ghép từ ảnh hoặc FPS giả định”** trước khi bấm phân tích để PDF mới cũng ghi trạng thái kỹ thuật.

Clip này ghép từ các ảnh validation, không có FPS và track ID nguồn đã xác minh. Không dùng số liệu từ nó để đánh giá một buổi tập. Nếu muốn kiểm tra trên sân thật, cần video quay cố định có FPS đáng tin cậy và đủ khung hình nhìn thấy người/bóng. Để báo số mét/km/h, hiệu chỉnh sân bằng bốn cặp điểm trong `FIELD_CONFIG["calibration_points"]` và đối chiếu kết quả với phép đo tham chiếu.

## Kiểm thử tự động và xử lý lỗi thường gặp

```powershell
python -m pytest -q -m "not slow"
```

- `Chưa tìm thấy checkpoint`: kiểm tra `best.pt` theo bước 2, hoặc chọn model gốc để kiểm tra giao diện; model gốc không tách thủ môn/trọng tài.
- `WinError 4551` khi tải `torch_python.dll` trong `venv`: Windows Application Control đang chặn thư viện đó. Thoát `venv` bằng `deactivate`, rồi chạy `python -m streamlit run dashboard.py` với Python hệ thống đã được kiểm tra trên máy này. Không tắt chính sách bảo vệ Windows để chạy thử.
- `Video thiếu frame hoặc metadata FPS`: xuất lại video với FPS cố định và codec thông dụng, rồi tải lại.
- PDF không có quãng đường/tốc độ: đó là hành vi đúng khi chưa hiệu chỉnh sân hoặc clip được đánh dấu kiểm tra kỹ thuật.
- Track ID nhiều hơn số người: tracker có thể sinh ID mới sau che khuất. Báo cáo hiện không ghép danh tính.
- PDF ghi nhiều frame “chưa rõ”: kiểm tra phát hiện bóng và khoảng cách gán bóng trước khi diễn giải phần tương tác bóng.
- Sau khi cập nhật mã pipeline, **dừng hẳn tiến trình Streamlit rồi chạy lại**; tải lại trang trình duyệt không bảo đảm nạp lại các module Python đã import. Trong `output/dashboard_jobs/<mã_lượt>/dashboard_run.json`, bản mới ghi `ball_confidence_threshold: 0.25` và `ball_assignment_direct_detection_only: true`. Nếu thiếu hai trường này, báo cáo thuộc tiến trình cũ; hãy khởi động lại rồi phân tích lại video.
- Nếu bóng không được phát hiện hoặc không gán được cho đội nào, PDF và dashboard ghi **Không đo được** thay vì trình bày 0%/100% như tỷ lệ kiểm soát bóng. Ngưỡng bóng hiện là 0,25, riêng các lớp người vẫn là 0,5; cần đối chiếu video vì ngưỡng thấp có thể sinh phát hiện nhầm. Frame bóng nội suy chỉ hỗ trợ hiển thị, không tính là phát hiện trực tiếp và không dùng để gán bóng.
- Bảng PDF chỉ hiện track xuất hiện ít nhất 1 giây; CSV giữ mọi track. Số track không phải số người duy nhất, đặc biệt khi có người ngoài sân hoặc cầu thủ bị mất dấu.

Các file `output/`, `datasets/`, `runs/`, `stubs/` là dữ liệu hoặc kết quả cục bộ, được `.gitignore` loại khỏi commit. Repo GitHub chỉ chứa mã, tài liệu và số liệu/hình của thí nghiệm học thuật đã ghi nguồn.
