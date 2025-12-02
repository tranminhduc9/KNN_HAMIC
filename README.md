# HAMIC — KNN Project Plan

## Thành viên & Vai trò

| Thành viên | Vai trò chính | Tệp phụ trách | Ghi chú |
|---|---|---|---|
| <Thành viên 1> | EDA, tiền xử lý & trực quan hóa | `src/eda.ipynb` | Đọc dữ liệu từ `data/`, làm sạch, chuẩn hóa/biến đổi, lưu `cleaned_data` và xuất hình. |
| <Điền tên 2> | Xây dựng mô hình KNN (hàm dùng chung) | `src/knn_model.py` | Viết các hàm KNN (nhiều hệ đo: Manhattan, Euclidean, Minkowski, Chebyshev; mở rộng nếu cần), kèm chú thích rõ ràng. |
| <Điền tên 3> | Đánh giá & đo lường | `src/knn_metrics.ipynb` | Gọi các hàm từ phần KNN để đánh giá theo bài toán phân loại/hồi quy, chọn K/hệ đo tối ưu. |

## Nhiệm vụ chi tiết

### 1) EDA — Khám phá dữ liệu, làm sạch, trực quan hóa (`src/eda.ipynb`)
- Nguồn dữ liệu đầu vào: thư mục `data/` (ví dụ: `data/raw.csv` hoặc các file tương tự).
- Kết quả/đầu ra bắt buộc:
	- File dữ liệu sạch: `data/cleaned_data.csv` (hoặc `.parquet` nếu dữ liệu lớn), có ghi rõ các biến/kiểu dữ liệu.
	- Thư mục hình ảnh: `data/figures/` chứa các biểu đồ thăm dò (histogram, boxplot, pairplot, heatmap, countplot...).
- Công việc chính:
	- Kiểm tra và xử lý giá trị thiếu, ngoại lệ (outliers), lỗi mã hóa/định dạng; chuẩn hóa tên cột, kiểu dữ liệu.
	- Tiền xử lý phù hợp mô hình KNN: scale/standardize (MinMaxScaler/StandardScaler) cho đặc trưng số; encode đặc trưng phân loại (OneHot/Ordinal) nếu cần.
	- Phân tích phân phối biến, tương quan (corr heatmap), quan hệ đặc trưng–mục tiêu, cân bằng lớp (nếu phân loại).
	- Lưu toàn bộ giả định/quy tắc chuyển đổi (data dictionary ngắn) để đảm bảo tái lập.
- Tiêu chí hoàn thành:
	- `data/cleaned_data.*` tồn tại và có thể nạp trực tiếp cho mô hình KNN.
	- Bộ hình minh họa đủ để hiểu dữ liệu và lựa chọn tiền xử lý phù hợp.

### 2) KNN Model — Xây dựng các hàm KNN dùng chung (`src/knn_model.py`)
- Yêu cầu: xây dựng các hàm có chú thích/docstring rõ ràng để `knn_metrics.ipynb` có thể gọi trực tiếp.
- Danh sách hàm khuyến nghị (tên có thể điều chỉnh nhưng cần thống nhất):
	- `compute_distance(x1, x2, metric="euclidean", p=2, VI=None)`: hỗ trợ các hệ đo sau:
		- Manhattan (L1: `metric="manhattan"`), Euclidean (L2: `metric="euclidean"`), Minkowski (`metric="minkowski"`, tham số `p`), Chebyshev (`metric="chebyshev"`).
		- (Tùy chọn nâng cao) Mahalanobis (`metric="mahalanobis"`, dùng ma trận nghịch đảo hiệp phương sai `VI`).
	- `knn_predict_classification(X_train, y_train, X_test, k=5, metric="euclidean", weights="uniform", p=2)`: dự đoán nhãn; `weights` nhận `"uniform"` hoặc `"distance"` (trọng số nghịch khoảng cách).
	- `knn_predict_proba(X_train, y_train, X_test, k=5, metric="euclidean", weights="uniform", p=2)`: trả về xác suất lớp (nếu cần cho ROC-AUC).
	- `knn_predict_regression(X_train, y_train, X_test, k=5, metric="euclidean", weights="uniform", p=2)`: dự đoán giá trị số (bình quân có/không trọng số).
- Quy ước/ghi chú triển khai:
	- Xử lý rõ ràng khi số hàng xóm bằng nhau trong phân loại (ví dụ: ưu tiên lớp có tổng trọng số lớn hơn, hoặc theo thứ tự lớp đã quy ước, cần ghi rõ).
	- Đầu vào giả định đã được chuẩn hóa/encode từ bước EDA. Nếu cần, bổ sung util `ensure_numeric(X)` để kiểm tra.
	- Viết tests nhỏ trong notebook để kiểm chứng hàm hoạt động với dữ liệu.

### 3) Metrics & Chọn K/hệ đo tối ưu (`src/knn_metrics.ipynb`)
- Xác định loại bài toán dựa trên `y`:
	- Phân loại: `y` rời rạc (nhãn), có thể nhị phân hoặc đa lớp.
	- Hồi quy: `y` liên tục (số thực).
- Độ đo đánh giá:
	- Phân loại: Accuracy, Precision, Recall, F1-score; Confusion Matrix; (nhị phân) ROC-AUC, PR-AUC khi cần class imbalance.
	- Hồi quy: MAE, RMSE, R².
- Chọn K và hệ đo (metric) tối ưu:
	- Sử dụng k-fold cross-validation (ví dụ k=5) trên lưới tham số: `K ∈ {1,3,5,...,49}` (có thể điều chỉnh) và `metric ∈ {euclidean, manhattan, minkowski(p), chebyshev}`.
	- Với Minkowski, thử một vài `p` đặc trưng (ví dụ p∈{1.5, 2, 3}).
	- Tiêu chí chọn: tối đa hóa độ đo chính (ví dụ: F1 hoặc Accuracy cho phân loại; RMSE tối thiểu hoặc R² tối đa cho hồi quy). Nếu hòa, ưu tiên K nhỏ hơn (mô hình đơn giản hơn).
- Báo cáo trực quan:
	- Biểu đồ hiệu năng theo K cho từng hệ đo; so sánh các metric.
	- (Phân loại) Confusion matrix trên tập validation/test với cấu hình tối ưu.
	- (Hồi quy) Biểu đồ dự đoán vs thực tế, residual plot.
- Xuất kết quả:
	- Ghi `best_config.json` (ví dụ: `{ "K": 11, "metric": "manhattan", "weights": "distance" }`).
	- Lưu các đồ thị vào `data/figures/` cùng tên dễ nhận biết.

### 4) Hiểu dữ liệu & áp dụng đúng bài toán
- Rà soát cột mục tiêu để quyết định phân loại hay hồi quy; ghi rõ trong notebook.
- Chọn độ đo đánh giá tương ứng (mục ở trên) và lý giải tại sao phù hợp với dữ liệu (ví dụ: dữ liệu mất cân bằng → ưu tiên F1/PR-AUC hơn Accuracy).
- Đánh giá K tối ưu và hệ đo tối ưu bằng quy trình cross-validation; nêu rõ tiêu chí quyết định.

## Cấu trúc thư mục (đề xuất)
```
data/
	raw.*                # dữ liệu gốc
	cleaned_data.*       # dữ liệu sạch cho mô hình
	figures/             # biểu đồ EDA & đánh giá
src/
	eda.ipynb
	knn_model.py
	knn_metrics.ipynb
```


