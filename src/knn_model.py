import numpy as np


def ensure_numeric(X):
    """
    Đảm bảo dữ liệu đầu vào là mảng số (numeric).

    Tham số
    --------
    X : array-like
        Dữ liệu đầu vào (sau khi đã encode nếu có).

    Trả về
    -------
    numpy.ndarray
        Mảng NumPy chứa dữ liệu dạng số.
    """
    X = np.array(X)
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("Dữ liệu đầu vào phải là dạng số (sau khi encode).")
    return X


def compute_distance(x1, x2, metric="euclidean", p=2, VI=None):
    """
    Tính khoảng cách giữa hai vector theo hệ đo được chọn.

    Tham số
    --------
    x1, x2 : array-like
        Hai vector đầu vào.
    metric : str, mặc định = "euclidean"
        Hệ đo khoảng cách:
        - "euclidean"   : Khoảng cách Euclid (L2)
        - "manhattan"  : Khoảng cách Manhattan (L1)
        - "minkowski"  : Khoảng cách Minkowski (tham số p)
        - "chebyshev"  : Khoảng cách Chebyshev
        - "mahalanobis": Khoảng cách Mahalanobis (nâng cao)
    p : int hoặc float, mặc định = 2
        Tham số p cho hệ đo Minkowski.
    VI : array-like, mặc định = None
        Ma trận nghịch đảo hiệp phương sai dùng cho Mahalanobis.

    Trả về
    -------
    float
        Giá trị khoảng cách giữa x1 và x2.
    """
    x1 = np.array(x1)
    x2 = np.array(x2)

    if metric == "euclidean":
        return np.sqrt(np.sum((x1 - x2) ** 2))

    elif metric == "manhattan":
        return np.sum(np.abs(x1 - x2))

    elif metric == "minkowski":
        return np.power(np.sum(np.abs(x1 - x2) ** p), 1 / p)

    elif metric == "chebyshev":
        return np.max(np.abs(x1 - x2))

    elif metric == "mahalanobis":
        if VI is None:
            raise ValueError("Mahalanobis cần cung cấp ma trận VI.")
        diff = x1 - x2
        return np.sqrt(diff.T @ VI @ diff)

    else:
        raise ValueError(f"Hệ đo không được hỗ trợ: {metric}")


def knn_predict_classification(X_train, y_train, X_test, k=5,
                                metric="euclidean", weights="uniform", p=2):
    """
    Dự đoán nhãn cho bài toán phân loại bằng thuật toán KNN.

    Quy tắc xử lý hòa phiếu:
    - Nếu dùng trọng số, chọn lớp có tổng trọng số lớn hơn.
    - Nếu vẫn hòa, chọn lớp có nhãn nhỏ hơn.

    Tham số
    --------
    X_train : array-like, shape (n_samples, n_features)
        Dữ liệu huấn luyện.
    y_train : array-like, shape (n_samples,)
        Nhãn huấn luyện.
    X_test : array-like, shape (m_samples, n_features)
        Dữ liệu cần dự đoán.
    k : int, mặc định = 5
        Số hàng xóm gần nhất.
    metric : str, mặc định = "euclidean"
        Hệ đo khoảng cách.
    weights : str, mặc định = "uniform"
        - "uniform"  : các hàng xóm có trọng số bằng nhau
        - "distance" : trọng số tỉ lệ nghịch với khoảng cách
    p : int hoặc float, mặc định = 2
        Tham số cho Minkowski.

    Trả về
    -------
    numpy.ndarray
        Mảng nhãn dự đoán cho X_test.
    """
    X_train = ensure_numeric(X_train)
    X_test = ensure_numeric(X_test)
    y_train = np.array(y_train)

    n_train = X_train.shape[0]
    k = min(k, n_train)

    y_pred = []
    eps = 1e-9  # tránh chia cho 0

    for x_test in X_test:
        distances = []

        for x_train, y in zip(X_train, y_train):
            d = compute_distance(x_test, x_train, metric=metric, p=p)
            distances.append((d, y))

        # Sắp xếp theo khoảng cách tăng dần
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        class_scores = {}

        for d, label in neighbors:
            if weights == "uniform":
                w = 1.0
            elif weights == "distance":
                w = 1.0 / (d + eps)
            else:
                raise ValueError("weights phải là 'uniform' hoặc 'distance'")

            class_scores[label] = class_scores.get(label, 0) + w

        max_score = max(class_scores.values())
        best_classes = [cls for cls, score in class_scores.items() if score == max_score]

        # Xử lý hòa phiếu: chọn nhãn nhỏ nhất
        predicted_class = sorted(best_classes)[0]
        y_pred.append(predicted_class)

    return np.array(y_pred)


def knn_predict_proba(X_train, y_train, X_test, k=5,
                      metric="euclidean", weights="uniform", p=2):
    """
    Dự đoán xác suất của các lớp bằng thuật toán KNN.

    Dùng để tính ROC-AUC hoặc PR-AUC trong bài toán phân loại.

    Tham số
    --------
    Giống như knn_predict_classification.

    Trả về
    -------
    numpy.ndarray
        Ma trận xác suất có dạng (m_samples, n_classes).
    """
    X_train = ensure_numeric(X_train)
    X_test = ensure_numeric(X_test)
    y_train = np.array(y_train)

    classes = np.unique(y_train)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    n_train = X_train.shape[0]
    k = min(k, n_train)

    eps = 1e-9
    probas = []

    for x_test in X_test:
        distances = []

        for x_train, y in zip(X_train, y_train):
            d = compute_distance(x_test, x_train, metric=metric, p=p)
            distances.append((d, y))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        class_scores = np.zeros(len(classes))

        for d, label in neighbors:
            if weights == "uniform":
                w = 1.0
            elif weights == "distance":
                w = 1.0 / (d + eps)
            else:
                raise ValueError("weights phải là 'uniform' hoặc 'distance'")

            class_scores[class_to_index[label]] += w

        total = np.sum(class_scores)
        probas.append(class_scores / total)

    return np.array(probas)


def knn_predict_regression(X_train, y_train, X_test, k=5,
                           metric="euclidean", weights="uniform", p=2):
    """
    Dự đoán giá trị cho bài toán hồi quy bằng thuật toán KNN.

    Tham số
    --------
    X_train : array-like
        Dữ liệu huấn luyện.
    y_train : array-like
        Giá trị đầu ra (liên tục).
    X_test : array-like
        Dữ liệu cần dự đoán.
    k : int, mặc định = 5
        Số hàng xóm gần nhất.
    metric : str
        Hệ đo khoảng cách.
    weights : str
        - "uniform"  : trung bình thường
        - "distance" : trung bình có trọng số nghịch khoảng cách
    p : int hoặc float
        Tham số Minkowski.

    Trả về
    -------
    numpy.ndarray
        Mảng giá trị dự đoán.
    """
    X_train = ensure_numeric(X_train)
    X_test = ensure_numeric(X_test)
    y_train = np.array(y_train, dtype=float)

    n_train = X_train.shape[0]
    k = min(k, n_train)

    y_pred = []
    eps = 1e-9

    for x_test in X_test:
        distances = []

        for x_train, y in zip(X_train, y_train):
            d = compute_distance(x_test, x_train, metric=metric, p=p)
            distances.append((d, y))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        if weights == "uniform":
            pred = np.mean([y for _, y in neighbors])

        elif weights == "distance":
            weights_vals = np.array([1.0 / (d + eps) for d, _ in neighbors])
            y_vals = np.array([y for _, y in neighbors])
            pred = np.sum(weights_vals * y_vals) / np.sum(weights_vals)

        else:
            raise ValueError("weights phải là 'uniform' hoặc 'distance'")

        y_pred.append(pred)

    return np.array(y_pred)
