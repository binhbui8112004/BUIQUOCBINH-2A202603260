# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không thay đổi gì, chạy đúng theo notebook.

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Record hạng 1: `class_id`= 568 (mình ước lượng theo biểu đồ, class của cab) ; `class_name`= cab ; `rank`=1 ; `score`= ~0.49 ; `taxonomy_name`= ImageNet-1K.
2. Record này nói về **cả bức ảnh**. Vì đây là bài toán Image Classification (file `yolo11n-cls.pt`), model chỉ đưa ra 1 dự đoán cho toàn bộ bức ảnh chứ không khoanh vùng từng vật thể.
3. Danh sách tên lớp do **đi kèm taxonomy/checkpoint**. Model không tự nghĩ ra tên, nó được huấn luyện trên bộ dữ liệu ImageNet-1K nên chỉ biết các class trong bộ này.
4. Cần ghi cả 3 vì: `class_id` để máy tính dễ xử lý, `class_name` để người đọc hiểu, và `taxonomy_name` để biết nhãn này thuộc bộ dữ liệu nào (tránh nhầm lẫn giữa các bộ dataset khác nhau).
5. Guideline cần nói rõ: Nếu ảnh có nhiều vật, phải chọn lớp của **vật thể chính, chiếm diện tích lớn nhất** hoặc là tâm điểm của bức ảnh.
6. Không được lấy `score` làm ground truth vì score chỉ là độ tự tin của model. Model có thể đoán sai (ví dụ ảnh toàn xe bus nhưng nó đoán là xe taxi với score cao), nhãn chuẩn phải do con người xác định.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Mình chọn record của người đàn ông (person): `class_name`= person ; `score`= 0.91 ; `bbox_xyxy`= [510, 80, 640, 410] (số này mình ước lượng trên ảnh vì file JSON không show hết) ; `bbox_width`= 130 ; `bbox_height`= 330.
2. Giải thích box: Góc trên-trái của hộp nằm ở tọa độ (510, 80), góc dưới-phải ở (640, 410). Chiều rộng của hộp là 130 pixel, chiều cao là 330 pixel. Hộp này bao trọn lấy người đàn ông đang đứng quay lưng.
3. Chọn hai ngưỡng 0.20 và 0.60: 
   - Ngưỡng 0.20 có 17 prediction.
   - Ngưỡng 0.60 có 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm việc nhiều hơn vì phải kiểm tra xem những hộp có score thấp có phải là vật thể thật không, hay chỉ là rác (false positive).
5. Quy tắc box: Luôn vẽ hộp bao sát vào phần hiển thị của vật thể. Nếu vật thể bị che khuất, chỉ vẽ hộp bao quanh phần nhìn thấy được, không tự đoán phần bị che.
6. Trên ảnh kitchen, cái lò nướng (oven) bên trái bị cắt mất một nửa mép ảnh. Mình sẽ dùng **escalation** (hỏi Lab Coach) vì trường hợp bị cắt mép thế này rất dễ gây tranh cãi, tùy vào guideline của từng dự án mà có thể yêu cầu vẽ hay bỏ qua.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Mình chọn record của người đàn ông: `instance_id`= kitchen-001 ; `class_name`= person ; `score`= 0.90 ; số điểm polygon `polygon_point_count`= rất nhiều (hàng trăm điểm) ; 3 điểm đầu của `polygon_xy`= [[496.0, 331.0], [496.0, 304.0], [495.0, 303.0], ...]
2. So với box (hình chữ nhật), polygon cho thêm thông tin chính xác về **hình dạng đường viền** của vật thể, giúp tách biệt vật thể với nền xung quanh tốt hơn.
3. `instance_id` dùng để phân biệt các vật thể khác nhau dù chúng cùng một class (ví dụ có 3 cái bát thì sẽ có 3 instance_id khác nhau). Nó không phải class ID (vì class ID chỉ nói đó là bát) và không phải tracking ID (vì tracking dùng để theo dõi vật thể di chuyển qua các frame video).
4. Quy tắc vẽ biên mask: Vẽ đường viền polygon men theo sát các pixel của vật thể, không để lấn sang vật thể bên cạnh hoặc lấn sang background.
5. Trên ảnh kitchen, mấy cái bát (bowl) nằm sát nhau trên bàn, viền của chúng bị dính vào nhau. Mình sẽ dùng **guideline** (nếu guideline có ghi rõ quy tắc tách viền) hoặc **escalation** (nếu guideline không nói rõ), vì đây là ca khó, ranh giới giữa các vật chạm nhau rất dễ vẽ sai.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Là 1 nhãn duy nhất cho toàn bộ ảnh (VD: cab). | Ảnh traffic có rất nhiều xe khác nhau, khó chọn 1 nhãn chính xác nếu không có guideline. | Xác định vật thể chính và gán nhãn tương ứng. | Kiểm tra xem nhãn có đúng là tâm điểm của bức ảnh không. |
| Phát hiện vật thể | Là các hộp (bbox) kèm nhãn class. | Ở ngưỡng 0.20, model đoán ra cả cái cốc, cái thìa (rất nhỏ) dễ bị nhầm là vật thể chính. | Vẽ hộp bao sát vật thể. | Kiểm tra xem có bỏ sót vật thể nào không, hộp vẽ có bị thừa/thiếu không. |
| Instance segmentation | Là các đa giác (polygon) bao quanh vật thể. | Các vật thể chạm nhau (như mấy cái bát) làm đường viền mask bị dính vào nhau. | Vẽ mask men theo viền vật thể. | Kiểm tra độ chính xác của các điểm ảnh trên đường viền. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu: Chỉ sử dụng 3 ảnh COCO công khai do bài lab cung cấp, tuyệt đối không tải ảnh cá nhân hay dữ liệu nội bộ lên môi trường Colab.
2. Nếu gặp ảnh/dữ liệu không đúng phạm vi: Mình sẽ dừng lại ngay và báo cáo cho Lab Coach/Giảng viên để xin hướng dẫn, không tự ý xử lý.

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS (Lưu ý: Bước này bạn phải tự làm trên LMS nha)