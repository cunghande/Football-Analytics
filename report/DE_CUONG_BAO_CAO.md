# Đề cương báo cáo bài tập lớn

## Trang đầu và tóm tắt

- Tên đề tài gợi ý: **Phát triển hệ thống phân tích video bóng đá bằng YOLO và ByteTrack: fine-tune detector, cải thiện chia đội và trực quan hóa**.
- Ghi tên môn, thành viên, giảng viên và thời gian thực hiện theo yêu cầu thực tế của lớp.
- Tóm tắt 150–250 từ: mục tiêu, dữ liệu `detection-player` v4, ba thiết lập detector, cải tiến chia đội, CSV/heatmap/minimap/dashboard, kết quả chính, giới hạn.

## 1. Đặt vấn đề

1. Nhu cầu trích xuất người, bóng, đội và thống kê từ video bóng đá một camera.
2. Hạn chế khi dùng model COCO tổng quát: không có lớp thủ môn/trọng tài riêng trong quy trình gốc; ball là vật thể nhỏ.
3. Mục tiêu: bốn lớp `ball, goalkeeper, player, referee`; so sánh ba cách fine-tune; chia đội ổn định hơn; xuất và xem kết quả.
4. Phạm vi: một camera, không re-ID, không tự hiệu chỉnh sân; pass map ứng viên là phần tùy chọn, nhận diện đội hình chưa triển khai.

## 2. Cơ sở lý thuyết

1. Phát hiện vật thể YOLO, pretrained và transfer learning; đóng băng `freeze=10` so với fine-tune toàn bộ. Mô tả metric precision, recall, mAP@0.5, mAP@0.5:0.95 và AP riêng ball.
2. ByteTrack và tính liên tục của track ID; mất ID ảnh hưởng thống kê theo cầu thủ.
3. KMeans trên màu áo, vùng thân áo, không gian Lab, bỏ phiếu theo thời gian và phân cụm lại. Nhấn mạnh cluster ID không phải tên đội.
4. Quy tắc gán bóng theo khoảng cách, quãng đường Euclid giữa frame liên tiếp, tốc độ theo FPS; phép đổi sang mét cần hiệu chỉnh sân.
5. Heatmap là mật độ lần quan sát; minimap chế độ sân hiệu chỉnh hoặc tọa độ ảnh.

## 3. Phân tích dự án mã nguồn gốc và dữ liệu

- Trích nguồn repo [alina-letzien/football-analytics](https://github.com/alina-letzien/football-analytics), giấy phép MIT. Ghi bảng thành phần **kế thừa** (pipeline video, YOLO/ByteTrack, KMeans, gán bóng, ước tính chuyển động) và **đóng góp** (fine-tune, thí nghiệm, chia đội theo thời gian, tách role, CSV, trực quan hóa, Streamlit).
- Dữ liệu [FSDM `detection-player` v4](https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4), CC BY 4.0. Nêu bản xuất YOLOv8, bốn lớp, kiểm tra nhãn, split theo clip và nguy cơ rò rỉ cùng trận. Nguồn chi tiết: [hướng dẫn dataset](../docs/DATASET_GUIDE.md), [kiểm kê dataset](../docs/DATASET_AUDIT_V4.md), [báo cáo A](BAO_CAO_BUOC_A.md).
- Chèn [biểu đồ phân bố dữ liệu](figures/dataset_distribution.png) và ảnh nhãn ví dụ từ báo cáo A. Chú thích rõ ảnh augment và ảnh gốc nếu thống kê cả hai.

## 4. Thiết kế cải tiến

1. Sơ đồ hệ thống: upload/video → YOLO bốn lớp → ByteTrack theo role → chia đội → bóng/chuyển động → CSV và heatmap/minimap → Streamlit.
2. Thiết lập train trên Modal: cùng dữ liệu, checkpoint gốc, `freeze=10`, fine-tune toàn bộ; nêu seed, epoch, kích thước ảnh, cấu hình GPU và tiêu chí chọn best checkpoint từ [báo cáo A](BAO_CAO_BUOC_A.md), [B](BAO_CAO_BUOC_B.md).
3. Chia đội cải tiến: vùng áo, lọc xanh sân có thể tắt, đặc trưng Lab, ngưỡng unknown, bỏ phiếu 7 frame, tùy chọn cập nhật cụm; thủ môn/trọng tài đi luồng riêng. Tham chiếu [báo cáo C](BAO_CAO_BUOC_C.md).
4. Định nghĩa CSV và mẫu số % gán bóng từ [báo cáo D](BAO_CAO_BUOC_D.md); chế độ không hiệu chỉnh chỉ là **ước tính**.
5. Định nghĩa heatmap/minimap từ [báo cáo E](BAO_CAO_BUOC_E.md); giao diện Streamlit và giới hạn upload từ [báo cáo F](BAO_CAO_BUOC_F.md).
6. Pass map ứng viên từ [báo cáo G](BAO_CAO_BUOC_G.md): chỉ khi hai chủ bóng ổn định cùng đội có khoảng bóng tự do được quan sát, không gọi là pass đã xác minh.

## 5. Thực nghiệm và đánh giá

### 5.1. Phát hiện bốn lớp

Trình bày giao thức so sánh trên cùng **599 frame validation**, cùng lớp và cách ánh xạ baseline COCO (`person→player`, `sports ball→ball`). Bảng bên dưới là số đo đã có từ [báo cáo B](BAO_CAO_BUOC_B.md), không thay bằng log train riêng của từng run:

| Thiết lập | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 ball |
|---|---:|---:|---:|---:|---:|
| Model gốc | 0,22781 | 0,24673 | 0,18948 | 0,06199 | 0,11348 |
| Freeze=10 | **0,59359** | **0,57800** | **0,55910** | **0,27091** | **0,44256** |
| Fine-tune toàn bộ | 0,58154 | 0,56304 | 0,52573 | 0,24734 | 0,40077 |

Chèn [biểu đồ so sánh](figures/comparison_b_metrics.png), [AP theo lớp](figures/comparison_b_per_class.png) và [chi tiết ball](figures/ball_detail.png). Diễn giải rằng `freeze=10` cao nhất **trên validation này**, chưa đủ kết luận tổng quát cho mọi trận; tập split theo clip nhưng chưa xác minh độc lập theo trận.

### 5.2. Chia đội

Nêu 50 frame 148–197 liên tiếp, 200 crop cầu thủ đã xem thủ công, dùng **box GT** để cô lập tác vụ chia đội. Bảng từ [báo cáo C](BAO_CAO_BUOC_C.md):

| Thiết lập | Mẫu đúng | Tỷ lệ |
|---|---:|---:|
| Mã gốc | 116/200 | 58,0% |
| Vùng áo + Lab + bỏ phiếu, không phân cụm lại | **194/200** | **97,0%** |
| Thêm phân cụm lại | 189/200 | 94,5% |

Chèn [biểu đồ ablation](figures/team_assignment_comparison.png) và [heatmap theo frame](figures/team_assignment_frame_heatmap.png). Nói rõ phân cụm lại giảm 5 mẫu đúng trong clip này. Các frame liên tiếp và mẫu cố ý chọn không phải quan sát độc lập; không gọi đây là độ chính xác end-to-end.

### 5.3. Chức năng và minh họa

- Mô tả kiểm thử CSV bằng đường đi/tốc độ giả lập có đáp án biết trước, không đưa số giả lập vào bảng kết quả bóng đá.
- Chèn [hình mật độ 50 frame có box GT](figures/dataset_spatial_preview_50frames.png): 594 box player, tọa độ ảnh, không có ID đội hay đường chạy.
- Chèn [frame smoke test pipeline](figures/dashboard_pipeline_smoke.png), ghi **ba frame ghép thành clip 10 FPS giả định chỉ để kiểm thử phần mềm**. Không lấy tốc độ, quãng đường hoặc % bóng của clip này làm kết quả thực nghiệm.
- Chèn [pass map giả lập](figures/pass_map_schematic_pass_map.png) nếu trình bày bước G; ghi rõ đây là ví dụ giải thích quy tắc, không phải số chuyền từ trận thật.
- Khi có video demo thật: bổ sung ảnh đầu vào/đầu ra cùng frame, bảng CSV tóm tắt, biểu đồ gán bóng, heatmap; kiểm tra thủ công ball/team/tracks; ghi FPS, độ phân giải, thời lượng, checkpoint, trạng thái hiệu chỉnh sân.

## 6. Thảo luận hạn chế

- Nhãn bóng nhỏ, che khuất, camera đổi góc, ID switch, đồng phục gần màu, thủ môn không được gán đội, luật gán bóng chỉ dựa khoảng cách.
- Không có video demo và không có nhãn GT theo thời gian, nên chưa đo sai số tốc độ/quãng đường hoặc kiểm soát bóng. `0,1 m/pixel` mặc định không phải hiệu chỉnh hình học.
- Pass map chỉ có đường chuyền ứng viên; chưa có nhãn sự kiện để tính precision/recall, và có thể nhầm cú sút, phá bóng, đổi ID hoặc chuyển camera.
- Báo cáo riêng rõ những phần có **đánh giá định lượng** (detector, chia đội có điều kiện trên GT) và những phần mới có **kiểm thử chức năng** (CSV, heatmap, dashboard).

## 7. Kết luận và hướng phát triển

- Tóm tắt đóng góp định lượng: fine-tune cải thiện detector trên tập validation đã chọn; chia đội dùng vùng áo/Lab/bỏ phiếu tốt hơn mã gốc trong 50 frame kiểm tra.
- Tóm tắt chức năng: xuất CSV, heatmap/minimap với nhãn chế độ tọa độ, dashboard Streamlit tải video và báo cáo, pass map ứng viên chưa xác minh.
- Hướng tiếp: thu 100–300 frame tự gán nhãn từ video demo, đánh giá trên trận độc lập, hiệu chỉnh sân thủ công để ước tính mét, gán nhãn pass và kiểm tra sai số; nhận diện đội hình chỉ nếu đủ dữ liệu và thời gian.

## Tài liệu tham khảo và phụ lục

- Repo MIT gốc, dataset CC BY 4.0, tài liệu Ultralytics, ByteTrack, scikit-learn, OpenCV, Streamlit. Chỉ ghi tên paper nếu đọc và xác nhận chính xác bản gốc; không mặc định repo có paper riêng.
- Phụ lục: lệnh Modal, cấu hình train, danh sách file CSV, metadata của hình, nguồn ảnh/clip, phần mềm và phần cứng thực tế, commit repo dùng để nộp.
