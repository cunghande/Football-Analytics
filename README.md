# Football Analytics

Hệ thống phân tích video bóng đá bằng thị giác máy tính, phát triển từ [alina-letzien/football-analytics](https://github.com/alina-letzien/football-analytics) theo giấy phép MIT. Luồng chính sử dụng YOLO, ByteTrack, phân cụm màu áo và quy tắc gán bóng; đầu ra gồm video có chú thích, dữ liệu CSV, heatmap, minimap và báo cáo PDF theo track.

## Trạng thái hiện tại

- Detector mặc định là `detection_player_role_ft_full_v1/weights/best.pt`, gồm bốn lớp `ball`, `goalkeeper`, `player`, `referee`. Model được fine-tune trên Modal; dashboard suy luận **trên máy đang chạy dashboard**.
- Dashboard có một luồng phân tích. Khi Python nhận CUDA, YOLO dùng GPU 0; đọc video, tracking và tạo báo cáo vẫn dùng CPU. Không có GPU, chạy trực tiếp Streamlit sẽ dùng CPU.
- Tải video qua trình duyệt tối đa **500 MB**, hoặc dùng đường dẫn tới tệp trên máy chạy ứng dụng. Mỗi lần phân tích tối đa **900 frame liên tiếp**; độ phân giải tự giảm theo ngân sách bộ nhớ. Báo cáo chỉ mô tả đoạn thực sự được phân tích.
- `P`, `GK`, `R` lần lượt là track cầu thủ, thủ môn và trọng tài. Track ID là mã tạm thời, không phải số áo hoặc danh tính duy nhất; một người có thể tạo nhiều track.
- PDF chỉ mô tả chỉ số, không chấm điểm cầu thủ. Khi chưa hiệu chỉnh sân, đơn vị là pixel và px/s; không báo khoảng cách theo mét hay tốc độ km/h. Đường chuyền trong báo cáo là **ứng viên chưa xác minh**.

## Chạy dashboard trên Windows

Trên máy dự án đã cấu hình Python 3.14 và GPU CUDA, mở PowerShell tại thư mục repo:

```powershell
cd D:\Code\football-analytics
powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
```

Mở `http://127.0.0.1:8501`, chọn video hoặc đường dẫn trên máy, chọn giây bắt đầu và bấm **Phân tích video**. Kiểm tra dòng thiết bị suy luận trong giao diện. Khi cập nhật mã, hãy dừng Streamlit bằng `Ctrl+C`, khởi động lại và chạy lại video để tạo báo cáo mới.

Script `run_dashboard.ps1` dành cho cấu hình Windows của máy này: nó chọn Python 3.14, yêu cầu CUDA và cổng 8501 trống. Với môi trường Python đã cài đủ dependency khác, có thể chạy:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

Lệnh trực tiếp chọn GPU CUDA nếu Python nhận được GPU; nếu không, ứng dụng dùng CPU. Hướng dẫn chi tiết, cách lấy checkpoint và xử lý lỗi nằm ở [hướng dẫn chạy và đọc báo cáo](docs/HUONG_DAN_TEST_BAO_CAO.md).

### Checkpoint

Dashboard yêu cầu tệp cục bộ:

```text
runs/modal/detection_player_role_ft_full_v1/weights/best.pt
```

Kiểm tra bằng `Test-Path .\runs\modal\detection_player_role_ft_full_v1\weights\best.pt`. Nếu thiết lập máy khác, lấy checkpoint từ Modal Volume `football-training-runs` theo hướng dẫn ở trên. Checkpoint và dataset không nằm trong GitHub; đừng nhầm mã nguồn với trọng số model đã huấn luyện.

## Kết quả và cách đọc

Mỗi lượt xử lý được lưu tại `output/dashboard_jobs/<mã-lượt>/`. Dashboard cho xem video và tải `analysis_session_report.pdf` hoặc ZIP báo cáo. ZIP gồm CSV theo track/đội, timeline gán bóng, hình, JSON phương pháp và ứng viên chuyền; video không nằm trong ZIP. `dashboard_run.json` ghi model, thiết bị, ngưỡng và đoạn đã xử lý.

Nhóm đội 1/2 là ID cụm màu áo, không tự mang tên đội. Thủ môn chưa được tự gán về đội một cách đáng tin cậy. Bóng có thể bị mất dấu; bóng nội suy hỗ trợ hiển thị nhưng không được tính là phát hiện trực tiếp để gán bóng. Các quy tắc ổn định vai trò và lọc người ngoài sân là ước đoán theo ảnh; cần kiểm tra video vì chúng vẫn có thể nhầm.

## Cấu trúc dự án

```text
dashboard.py                 Giao diện Streamlit
main.py                      Điều phối pipeline phân tích
config.py                    Model, ngưỡng và cấu hình phân tích
src/                         Detector, tracker, chia đội, CSV, PDF và hình
tests/                       Kiểm thử đơn vị và tích hợp
docs/                        Hướng dẫn sử dụng, dữ liệu và huấn luyện
report/                      Kết quả thực nghiệm và dữ liệu/hình hỗ trợ báo cáo
scripts/                     Audit dữ liệu, tái tạo thí nghiệm và hình
train.py, modal_train.py     Huấn luyện YOLO cục bộ hoặc trên Modal
modal_evaluate.py            Đánh giá checkpoint trên Modal
```

`datasets/`, `runs/`, `output/`, `stubs/` và `venv/` là dữ liệu hoặc kết quả cục bộ, không nằm trong Git. Các bản báo cáo Word và file khóa Word được giữ cục bộ để tránh đưa thông tin cá nhân lên repository.

## Dữ liệu và thí nghiệm

Nguồn dữ liệu chính: [FSDM detection-player v4](https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4), CC BY 4.0. Bản dẫn xuất đã làm sạch gồm 1.247 ảnh train, 599 validation và 244 test. Dữ liệu được chia theo tên clip, **chưa chứng minh độc lập theo trận**. Bộ sửa vai trò gồm 60 frame gán nhãn bổ sung chỉ được thêm vào train.

Các kết quả và giao thức:

- [Hướng dẫn dataset](docs/DATASET_GUIDE.md), [audit dữ liệu](docs/DATASET_AUDIT_V4.md), [huấn luyện Modal](docs/MODAL_TRAINING.md).
- [So sánh baseline, freeze=10 và fine-tune toàn bộ](report/BAO_CAO_BUOC_B.md); [đánh giá chia đội](report/BAO_CAO_BUOC_C.md).
- [Fine-tune bổ sung cho vai trò](report/BAO_CAO_FINE_TUNE_VAI_TRO.md), gồm cả kết quả trên validation/test và kiểm tra riêng 60 frame đã dùng train. Không coi kết quả trên 60 frame train là khả năng tổng quát.
- [Phân tích chuyển động và đường chuyền](report/PHAN_TICH_CHUYEN_DONG_VA_CHUYEN_BONG.md). Các báo cáo bước A–G còn lại là **nhật ký thực nghiệm**, có thể mô tả cấu hình tại thời điểm cũ; không phải hướng dẫn chạy phiên bản hiện tại.

Trên 599 ảnh validation theo giao thức COCOeval, model vai trò mới so với `freeze=10`: AP50 thủ môn `0,1775 → 0,2271`; mAP50 `0,5591 → 0,5501`; AP50 bóng `0,4426 → 0,3629`. Chất lượng không tăng đồng đều giữa các lớp; cần đánh giá thêm trên các trận có nhãn độc lập.

## Kiểm thử

```powershell
python -m pytest -q -m "not slow"
```

Kiểm thử chậm `python -m pytest -q -m slow` chạy `main.py` trên video mẫu nếu có, có thể mất nhiều phút và ghi đè kết quả mặc định trong `output/`. `main.py` có thể dùng cache trong `stubs/`; khi đổi video/model hoặc cần kiểm tra detector mới, đặt `VIDEO_CONFIG["read_from_stub"] = False` trong `config.py` hoặc dùng dashboard, vì dashboard tạo cache riêng và không đọc stub cũ. Không cần xóa dữ liệu cũ để chạy dashboard.

## Giấy phép và ghi nguồn

Mã dự án kế thừa [football-analytics](https://github.com/alina-letzien/football-analytics), MIT; xem [LICENSE](LICENSE). Dataset có giấy phép riêng CC BY 4.0, cần ghi FSDM, tên và phiên bản dataset, URL cùng các thay đổi dữ liệu. Tư liệu 60 frame tự gán nhãn được xuất từ workspace riêng tư, không công bố trong repo.
