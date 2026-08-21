import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        mlflow.log_params(params)

        model_type = params.get("model_type", "random_forest")
        if model_type == "random_forest":
            allowed = {"n_estimators", "max_depth", "min_samples_split", "max_features", "class_weight"}
            model_params = {k: v for k, v in params.items() if k in allowed}
            model = RandomForestClassifier(**model_params, random_state=42)
        elif model_type == "gradient_boosting":
            allowed = {"n_estimators", "max_depth", "min_samples_split", "learning_rate", "subsample"}
            model_params = {k: v for k, v in params.items() if k in allowed}
            model = GradientBoostingClassifier(**model_params, random_state=42)
        elif model_type == "logistic_regression":
            allowed = {"C", "solver", "penalty", "class_weight", "max_iter"}
            model_params = {k: v for k, v in params.items() if k in allowed}
            model_params.setdefault("max_iter", 2000)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(**model_params, random_state=42),
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("model_type", model_type)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        label_distribution = {
            str(label): float((y_train == label).mean())
            for label in sorted(y_train.unique())
        }
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
            "label_distribution": label_distribution,
        }
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open("outputs/report.txt", "w") as f:
            f.write(f"model_type: {model_type}\n")
            f.write(f"accuracy: {acc:.4f}\n")
            f.write(f"f1_score: {f1:.4f}\n\n")
            f.write("confusion_matrix:\n")
            f.write(f"{confusion_matrix(y_eval, preds).tolist()}\n\n")
            f.write("classification_report:\n")
            f.write(classification_report(y_eval, preds, zero_division=0))
            f.write("\ntraining_label_distribution:\n")
            f.write(json.dumps(label_distribution, indent=2))
            if any(ratio < 0.10 for ratio in label_distribution.values()):
                f.write("\nWARNING: at least one label represents less than 10% of training data.\n")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
