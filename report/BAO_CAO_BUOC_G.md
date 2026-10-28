# Bước G (tùy chọn) — Pass map từ đường chuyền ứng viên

## 1. Phạm vi

Chọn **pass map** thay cho nhận diện đội hình vì pipeline đã có track bóng và chuỗi cầu thủ được gán bóng theo frame. Module mới trong `src/pass_map.py` tạo **đường chuyền ứng viên**, không xác nhận sự kiện chuyền thật. Dataset `detection-player` v4 không có nhãn chuyền bóng, ID cầu thủ bền vững hoặc video có FPS thật; vì vậy bước này **chưa có precision, recall, F1 hay số đường chuyền được xác nhận trên trận thật**.

## 2. Quy tắc phát hiện

Từ dãy ID đang được gán bóng, chỉ ghi một ứng viên khi có dạng:

```text
A, ...  →  không gán cho ai (ít nhất 1 frame, tối đa 1,5 s)  →  B, ...
```

Điều kiện mặc định:

1. A và B là **hai track ID khác nhau**; ít nhất một bên có 2 frame liên tiếp được gán bóng.
2. Team ID ở các frame ổn định của A và B bằng nhau và thuộc đội 1 hoặc 2; loại `unknown` và đối phương.
3. Bóng có detection thật (`conf > 0`) tại frame cuối của A và frame đầu của B. Nếu bóng mất dấu ở khoảng giữa, ứng viên được đánh dấu `candidate_unverified_sparse_ball`; nếu thấy bóng ở giữa, đánh dấu `candidate_unverified_visible_flight`. Bóng chỉ do nội suy ở hai đầu không thỏa điều kiện.
4. Khoảng cách giữa vị trí chân A khi nhả bóng và B khi nhận bóng nằm trong `[0,03; 0,75]` của tọa độ chuẩn hóa 2D. Đây là ngưỡng **thử nghiệm của dự án**, chưa được tối ưu trên dữ liệu có nhãn.
5. Chuyển ID trực tiếp không có frame bóng tự do bị loại để giảm nhầm lẫn do nhiều cầu thủ đứng gần bóng. Quy tắc này sẽ bỏ sót một số đường chuyền nhanh.

Đầu ra `output/analysis_pass_candidates.csv` có frame/thời điểm, team ID, ID A/B, tọa độ đầu/cuối chuẩn hóa, khoảng cách chuẩn hóa, số frame thấy bóng và mức bằng chứng trong `status`. Cột mét chỉ có giá trị khi đã hiệu chỉnh sân thủ công. `output/analysis_pass_map.png`/`.pdf` vẽ mũi tên có hướng và số ứng viên theo cặp track; `output/analysis_pass_map_metadata.json` ghi chế độ tọa độ, tham số và cảnh báo. Dashboard hiển thị bản đồ, bảng ứng viên và thêm chúng vào ZIP tải xuống. [Kiểm tra trên clip 750 frame](PHAN_TICH_CHUYEN_DONG_VA_CHUYEN_BONG.md) có 5 ứng viên chưa xác minh.

## 3. Hình minh họa quy tắc

![Pass map giả lập cho hai đội](figures/pass_map_schematic_pass_map.png)

**Chú thích đề xuất:** Sơ đồ **giả lập** gồm hai chuyển quyền gán bóng thỏa quy tắc, một cho mỗi đội; mũi tên P11→P22 và P33→P44 là ID track giả định. Nền xám cho biết tọa độ ảnh chưa hiệu chỉnh, không phải mặt sân. Hình dùng để giải thích thuật toán và kiểm tra xuất PNG/PDF, **không phải kết quả từ dataset/trận thật**. Có [PDF](figures/pass_map_schematic_pass_map.pdf), [CSV giả lập](figures/pass_map_schematic_pass_candidates.csv) và [metadata](figures/pass_map_schematic_pass_map_metadata.json). Tạo lại bằng `python scripts/make_pass_map_schematic.py`.

## 4. Đánh giá và giới hạn

Kiểm thử tự động có đáp án biết trước: chuỗi A–bóng tự do–B cùng đội sinh đúng một ứng viên; khác đội, không có khoảng trống hoặc bóng chỉ nội suy ở hai đầu không sinh sự kiện; chế độ sân hiệu chỉnh tính đúng khoảng cách mét trong fixture giả lập. Smoke test ba frame của dashboard quá ngắn để thỏa điều kiện nên **đúng ra tạo bản đồ rỗng**; việc tạo được tệp rỗng hợp lệ chỉ kiểm chứng pipeline, không chứng minh nhận diện chuyền thật.

Nguy cơ sai gồm ID switch, camera cắt cảnh, bóng bị che, gán bóng sai theo khoảng cách, cầu thủ chạm bóng nhưng không kiểm soát, cú sút/pha phá bóng bị coi là chuyền, và bỏ sót pass nhanh hay chuyền qua người ngoài khung hình. Không có re-ID; P11/P22 là ID track ngắn hạn. Không hiệu chỉnh sân thì mũi tên chỉ biểu thị vị trí **trên ảnh**, không biểu thị quãng đường chuyền theo mét.

Khi có video demo, nên gán nhãn thủ công các pha chuyển quyền bóng trên một đoạn riêng: frame bắt đầu/kết thúc, đội, người chuyền/người nhận, pass/không pass. Đối chiếu sự kiện dự đoán với nhãn theo một ngưỡng thời gian định trước để tính precision/recall và phân tích false positive/false negative; không dùng cùng đoạn đã chỉnh ngưỡng làm tập đánh giá. Chỉ sau bước đó mới đưa số lượng/độ chính xác đường chuyền vào bảng thực nghiệm.

Tài liệu liên quan: [PassNet — Automatic Pass Annotation from Soccer VideoStreams Based on Object Detection and LSTM](https://arxiv.org/abs/2007.06475) nghiên cứu nhận diện pass từ video; [Using Network Science to Analyze Football Passing Networks](https://arxiv.org/abs/1807.00534) thảo luận biểu diễn mạng chuyền. **Quy tắc ở repo này là heuristic tự thiết kế**, không phải triển khai lại mô hình trong hai paper trên.
