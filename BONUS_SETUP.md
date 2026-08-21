# Bonus setup

## Bonus 1 — MLflow remote trên DagsHub

Tạo một repository trên DagsHub và lấy thông tin remote MLflow. Thêm ba GitHub Secrets:

```text
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub-username>
MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```

Workflow đã đọc các secrets này. Khi được khai báo, các runs từ job Train sẽ được ghi vào DagsHub thay vì chỉ lưu local.

## Bonus 2–5 đã triển khai

- `model_type` hỗ trợ `random_forest`, `gradient_boosting`, `logistic_regression`.
- `outputs/report.txt` có confusion matrix, precision, recall và F1 theo lớp.
- `outputs/metrics.json` có phân phối nhãn; report cảnh báo lớp có tỷ lệ dưới 10%.
- Workflow so sánh accuracy mới với `models/latest/metrics.json` trước khi promote model.
