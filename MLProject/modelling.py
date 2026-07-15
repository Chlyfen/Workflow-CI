"""
modelling.py (versi CI)
Digunakan oleh MLflow Project (`mlflow run .`) di dalam workflow GitHub Actions
untuk melakukan re-training model secara otomatis ketika trigger dipantik.

Artefak (model + metrics) disimpan pada folder mlruns lokal di runner, lalu
di-upload sebagai GitHub Actions artifact oleh workflow CI (lihat
.github/workflows/ci.yml pada root Workflow-CI).
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/telco_churn_preprocessed.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    mlflow.set_experiment("Telco_Churn_CI")
    mlflow.sklearn.autolog()

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="ci_retrain"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(f"[INFO] Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"[INFO] Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"[INFO] Recall   : {recall_score(y_test, y_pred):.4f}")
        print(f"[INFO] F1-score : {f1_score(y_test, y_pred):.4f}")

    print("[INFO] Retraining CI selesai.")


if __name__ == "__main__":
    main()
