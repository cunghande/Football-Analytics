# Bước E — Heatmap và minimap 2D

## 1. Đầu vào, đầu ra và ý nghĩa

Đầu vào khi chạy video là các track đã có `position` (tâm bóng hoặc chân người), team ID ở từng frame, kích thước ảnh và trạng thái hiệu chỉnh sân. `SpatialVisualizer` thêm minimap vào góc dưới bên phải của video; đồng thời xuất `output/analysis_minimap_last.png`, `output/analysis_heatmaps.png`, `.pdf` và `output/analysis_heatmaps_metadata.json`. Heatmap gồm ba ô **đội 1, đội 2, chưa rõ đội** trên cùng thang màu. Đây là mật độ **lần quan sát** theo vị trí, không phải quỹ đạo, số người duy nhất hay quãng đường chạy.

Hai chế độ tọa độ được tách rõ:

| Chế độ | Tọa độ chấm trên minimap/heatmap | Cách diễn giải |
|---|---|---|
| Có bốn cặp điểm hiệu chỉnh thủ công | Vị trí sân sau homography, chia theo chiều dài/rộng sân | Mô hình sân 2D; phụ thuộc chất lượng hiệu chỉnh và bù camera |
| Chưa hiệu chỉnh | Vị trí trên ảnh chia theo chiều rộng/cao của frame | **Sơ đồ vị trí trên ảnh**; không suy ra vị trí thật trên sân |

Ở chế độ chưa hiệu chỉnh, nền minimap màu xám, không vẽ vạch sân và có chữ `IMAGE SPACE (UNCALIBRATED)`. Tọa độ ngoài vùng hợp lệ hoặc không hữu hạn bị loại; số bị loại ghi trong JSON. Vạch sân chỉ xuất hiện khi đã hiệu chỉnh. Bước này không làm tự hiệu chỉnh sân và không làm re-ID.

## 2. Cách tính heatmap

Với mỗi box `player` trong mỗi frame, lấy điểm chân (trung điểm cạnh dưới box). Đưa điểm về khoảng 0–1 theo kích thước ảnh hoặc kích thước sân đã hiệu chỉnh. Tăng 1 cho ô lưới 48×72 tại team ID của **frame đó**. Ma trận được làm mượt Gaussian với `sigma=1,2` ô lưới để nhìn rõ vùng tập trung; ba panel dùng cùng một giá trị màu lớn nhất, nên có thể so độ đậm giữa đội 1 và đội 2. Số quan sát và đơn vị được ghi ngay trên hình và trong JSON.

Nhiều frame liên tiếp của cùng một người đóng góp nhiều lần. Vùng sáng nghĩa là box cầu thủ xuất hiện nhiều tại vùng ảnh/sân đó. Kết quả bị ảnh hưởng bởi thời lượng nhìn thấy, góc máy, vùng hình bị cắt, mất dấu, lỗi chia đội và phân bố frame; không diễn giải là xác suất chiến thuật hay hiệu quả thi đấu. Thủ môn/trọng tài/bóng hiện được vẽ trên minimap nhưng **không tính vào heatmap cầu thủ** để không trộn role.

## 3. Hình minh họa có dữ liệu thật từ validation

![Nhãn GT, minimap tọa độ ảnh và mật độ vị trí của 594 box cầu thủ trên 50 frame](figures/dataset_spatial_preview_50frames.png)

**Chú thích đề xuất cho báo cáo:** Nhãn gốc của [FSDM, `detection-player` v4](https://universe.roboflow.com/fsdm-olss1/detection-player/dataset/4), giấy phép CC BY 4.0, nhóm clip 2 trong tập validation dẫn xuất, 50 frame 148–197, một bản xuất mỗi số frame. Bên trái là box GT ở frame 148; giữa là vị trí chuẩn hóa trên **ảnh** của các role được gán nhãn tại frame này; bên phải là 594 box `player` từ 50 frame. Ảnh không có nhãn đội, track ID hay hiệu chỉnh sân, nên mọi cầu thủ đều hiện màu xám trên minimap và một heatmap chung. Đây là hình kiểm tra cách trực quan hóa bằng dữ liệu thật; **không phải kết quả theo dõi chuyển động hoặc đo quãng đường**. Có [PDF để chèn báo cáo](figures/dataset_spatial_preview_50frames.pdf) và [metadata của hình](figures/dataset_spatial_preview_metadata.json).

## 4. Tái lập

```powershell
# Tạo lại hình minh họa từ 50 frame validation đã lưu cục bộ:
python scripts/make_dataset_spatial_preview.py

# Khi có video và checkpoint bốn lớp:
$env:YOLO_MODEL_PATH = "runs/modal/detection_player_freeze10_v1/weights/best.pt"
python main.py
```

Để dùng chế độ sân hiệu chỉnh, đặt `FIELD_CONFIG["calibration_points"]` trong `config.py` với `src` là bốn điểm ảnh và `dst` là bốn vị trí tương ứng trên sân theo mét. Thứ tự các cặp điểm phải khớp nhau và vị trí phải cùng mặt phẳng sân. Khi chưa có điểm hiệu chỉnh, cứ để `None`; hình và metadata sẽ ghi rõ chế độ tọa độ ảnh.

## 5. Kiểm thử và giới hạn

Kiểm thử đơn vị xác nhận tâm ảnh `(100, 50)` với ảnh `200×100` nằm ở giữa minimap, điểm sát biên không vượt ô lưới, điểm ngoài ảnh bị loại, hai quan sát cùng ô được cộng thành 2, và khi hiệu chỉnh thì mã dùng `position_transformed` thay cho `position`. Hình PNG/PDF đã được tạo từ 50 frame validation và kiểm tra trực quan. Chưa có video demo, nên chưa thể đánh giá overlay minimap theo thời gian trên một trận thật hoặc kiểm tra độ chính xác vị trí sân so với tọa độ chuẩn.
