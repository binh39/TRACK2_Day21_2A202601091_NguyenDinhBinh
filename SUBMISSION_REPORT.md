# Báo cáo Lab MLOps — Day 21

## 1. Mục tiêu và kiến trúc

Pipeline triển khai mô hình Wine Quality gồm: DVC quản lý dữ liệu trên Amazon S3, GitHub Actions thực hiện Unit Test → Train → Eval (accuracy ≥ 0.70) → Deploy, và FastAPI phục vụ dự đoán trên Amazon EC2.

## 2. Kết quả thực hiện

- S3 bucket: `mlops-vinai-545863790277-us-east-1-an` (region `us-east-1`).
- DVC remote: `s3://mlops-vinai-545863790277-us-east-1-an/dvc`.
- Dữ liệu đã version hóa và push: `train_phase1.csv`, `eval.csv`, `train_phase2.csv`; Bước 3 đã gộp dữ liệu huấn luyện từ 2998 lên 5996 mẫu.
- EC2 API: `http://44.200.221.174:8000`.
- `/health`: `{"status":"ok"}`.
- `/predict`: trả về dự đoán hợp lệ, ví dụ `{"prediction":0,"label":"thap"}`.
- Model đã upload: `s3://mlops-vinai-545863790277-us-east-1-an/models/latest/model.pkl`.

## 3. Thí nghiệm MLflow

Các runs được thực hiện trên tập huấn luyện 5996 mẫu và `eval.csv`:

| n_estimators | max_depth | min_samples_split | accuracy | f1_score |
|---:|---:|---:|---:|---:|
| 100 | 5 | 2 | 0.5800 | 0.5690 |
| 300 | 10 | 2 | 0.6680 | 0.6635 |
| 500 | null | 2 | 0.7460 | 0.7451 |

Bộ tham số cuối cùng được chọn vì đạt accuracy `0.7460`, vượt eval gate `0.70`.

## 4. CI/CD

Workflow thành công ở các job Unit Test, Train và Eval với commit `a37502b`. Job Deploy đã được kiểm tra thủ công trên EC2 và API hoạt động. Cần chạy lại workflow sau khi cập nhật đúng GitHub Secrets `VM_HOST`, `VM_USER`, `VM_SSH_KEY` để cả bốn job xanh.

## 5. Bằng chứng cần chụp màn hình

1. MLflow UI: bảng có ít nhất 3 runs và hai metrics `accuracy`, `f1_score`.
2. GitHub Actions: run `MLOps Pipeline`, hiển thị bốn job Unit Test, Train, Eval, Deploy màu xanh.
3. S3 Console: prefix `dvc/` và `models/latest/model.pkl`.
4. Terminal/API: kết quả `/health` và `/predict`.
5. GitHub commit `ee5faa0`: commit dữ liệu mới kích hoạt continual training.
