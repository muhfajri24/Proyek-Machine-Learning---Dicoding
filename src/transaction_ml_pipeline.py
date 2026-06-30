from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
import shutil

import joblib
import kagglehub
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.cluster import KElbowVisualizer

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUBMISSION_DIR = PROJECT_ROOT / "BMLP_Muhammad-Fajri"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_DATASET_PATH = DATA_RAW_DIR / "financial_transactions.csv"
PREPROCESSED_DATASET_PATH = DATA_PROCESSED_DIR / "transactions_preprocessed.csv"
CLUSTERED_DATASET_PATH = DATA_PROCESSED_DIR / "transactions_training_with_target.csv"
SUMMARY_PATH = REPORTS_DIR / "cluster_interpretation.md"
PROJECT_SUMMARY_PATH = REPORTS_DIR / "project_summary.json"
SOURCE_CLUSTER_NOTEBOOK = NOTEBOOKS_DIR / "clustering_submission.ipynb"
SOURCE_CLASSIFICATION_NOTEBOOK = NOTEBOOKS_DIR / "classification_submission.ipynb"
SUBMISSION_CLUSTER_NOTEBOOK = SUBMISSION_DIR / "[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb"
SUBMISSION_CLASSIFICATION_NOTEBOOK = SUBMISSION_DIR / "[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb"
SUBMISSION_CLUSTER_MODEL = SUBMISSION_DIR / "model_clustering.h5"
SUBMISSION_CLASSIFICATION_MODEL = SUBMISSION_DIR / "decision_tree_model.h5"
SUBMISSION_CLUSTER_DATA = SUBMISSION_DIR / "data_clustering.csv"
REQUIRED_SUBMISSION_ARTIFACTS = {
    "[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb",
    "[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb",
    "model_clustering.h5",
    "decision_tree_model.h5",
    "data_clustering.csv",
}

KAGGLE_DATASET_SLUG = "aryan208/financial-transactions-dataset-for-fraud-detection"
KAGGLE_FILENAME = "financial_fraud_detection_dataset.csv"
SAMPLE_SIZE = 30_000
RANDOM_STATE = 42
CLUSTER_RANGE = (2, 8)


def ensure_directories() -> None:
    for folder in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, SUBMISSION_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_submission_directory() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    for path in SUBMISSION_DIR.iterdir():
        if path.name in REQUIRED_SUBMISSION_ARTIFACTS:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def locate_kaggle_dataset() -> Path:
    if RAW_DATASET_PATH.exists():
        return RAW_DATASET_PATH

    explicit_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "aryan208" / "financial-transactions-dataset-for-fraud-detection" / "versions" / "1" / KAGGLE_FILENAME
    if explicit_path.exists():
        return explicit_path

    downloaded_root = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    dataset_path = downloaded_root / KAGGLE_FILENAME
    if not dataset_path.exists():
        csv_candidates = sorted(downloaded_root.rglob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"Tidak menemukan file CSV pada dataset {KAGGLE_DATASET_SLUG}")
        dataset_path = csv_candidates[0]
    return dataset_path


def stage_kaggle_dataset(sample_size: int = SAMPLE_SIZE) -> Path:
    source_path = locate_kaggle_dataset()
    staged_df = pd.read_csv(source_path, nrows=sample_size)

    renamed_df = staged_df.rename(
        columns={
            "transaction_id": "TransactionID",
            "timestamp": "TransactionDate",
            "sender_account": "AccountID",
            "receiver_account": "CounterpartyAccount",
            "amount": "TransactionAmount",
            "transaction_type": "TransactionType",
            "merchant_category": "MerchantCategory",
            "location": "Location",
            "device_used": "DeviceType",
            "is_fraud": "IsFraud",
            "fraud_type": "FraudType",
            "time_since_last_transaction": "TimeSinceLastTransaction",
            "spending_deviation_score": "SpendingDeviationScore",
            "velocity_score": "VelocityScore",
            "geo_anomaly_score": "GeoAnomalyScore",
            "payment_channel": "PaymentChannel",
            "ip_address": "IPAddress",
            "device_hash": "DeviceID",
        }
    )

    renamed_df.to_csv(RAW_DATASET_PATH, index=False)
    return RAW_DATASET_PATH


def load_dataset(dataset_path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def dataframe_info_text(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def preprocess_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, LabelEncoder], StandardScaler]:
    missing_before = df.isnull().sum()
    duplicates_before = pd.Series({"duplicate_rows": int(df.duplicated().sum())})

    prepared = df.copy()
    high_missing_columns = ["FraudType", "TimeSinceLastTransaction"]
    prepared = prepared.drop(columns=high_missing_columns, errors="ignore")
    cleaned = prepared.dropna().drop_duplicates().copy()

    columns_to_drop = [
        "TransactionID",
        "AccountID",
        "CounterpartyAccount",
        "DeviceID",
        "IPAddress",
        "TransactionDate",
        "IsFraud",
    ]
    cleaned = cleaned.drop(columns=columns_to_drop, errors="ignore")

    encoded = cleaned.copy()
    encoders: dict[str, LabelEncoder] = {}
    categorical_columns = encoded.select_dtypes(include=["object"]).columns.tolist()

    for column in categorical_columns:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(encoded)
    scaled_df = pd.DataFrame(scaled_values, columns=encoded.columns, index=encoded.index)

    diagnostics_df = pd.DataFrame(
        {
            "feature": list(missing_before.index) + ["duplicate_rows"],
            "missing_count": list(missing_before.values) + [int(duplicates_before["duplicate_rows"])],
        }
    )
    diagnostics_df.to_csv(REPORTS_DIR / "data_quality_checks.csv", index=False)

    encoded.to_csv(PREPROCESSED_DATASET_PATH, index=False)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    quality_summary = {
        "missing_before": missing_before,
        "duplicates_before": duplicates_before,
    }
    return cleaned, scaled_df, quality_summary, encoders, scaler


def find_best_cluster(scaled_df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    model = KMeans(random_state=RANDOM_STATE, n_init=20)
    model._estimator_type = "clusterer"
    visualizer = KElbowVisualizer(model, k=CLUSTER_RANGE, timings=False)
    visualizer.fit(scaled_df)
    visualizer.finalize()
    visualizer.fig.savefig(FIGURES_DIR / "elbow_method.png", dpi=180, bbox_inches="tight")
    plt.close(visualizer.fig)

    metrics: list[dict[str, float]] = []
    for cluster_count in range(CLUSTER_RANGE[0], CLUSTER_RANGE[1] + 1):
        km = KMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(scaled_df)
        metrics.append(
            {
                "cluster_count": cluster_count,
                "inertia": float(km.inertia_),
                "silhouette_score": float(silhouette_score(scaled_df, labels)),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(REPORTS_DIR / "cluster_metrics.csv", index=False)
    best_k = int(visualizer.elbow_value_ or metrics_df.sort_values("silhouette_score", ascending=False).iloc[0]["cluster_count"])
    return best_k, metrics_df


def build_clustering_model(scaled_df: pd.DataFrame, best_k: int) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(scaled_df)
    joblib.dump(model, MODELS_DIR / "model_clustering.joblib")
    joblib.dump(model, MODELS_DIR / "model_clustering.h5")
    joblib.dump(model, SUBMISSION_CLUSTER_MODEL)
    return model, labels


def save_cluster_outputs(cleaned_df: pd.DataFrame, encoded_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    labeled_raw = cleaned_df.copy()
    labeled_raw["Target"] = labels

    labeled_training = encoded_df.copy()
    labeled_training["Target"] = labels
    labeled_training.to_csv(CLUSTERED_DATASET_PATH, index=False)
    labeled_training.to_csv(SUBMISSION_CLUSTER_DATA, index=False)

    numeric_columns = labeled_raw.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [column for column in numeric_columns if column != "Target"]

    cluster_stats = labeled_raw.groupby("Target")[numeric_columns].agg(["mean", "min", "max"]).round(2)
    cluster_stats.to_csv(REPORTS_DIR / "cluster_numeric_summary.csv")

    aggregated = labeled_raw.groupby("Target", as_index=False).agg(
        transaction_count=("TransactionAmount", "count"),
        avg_transaction_amount=("TransactionAmount", "mean"),
        avg_spending_deviation=("SpendingDeviationScore", "mean"),
        avg_velocity_score=("VelocityScore", "mean"),
        avg_geo_anomaly_score=("GeoAnomalyScore", "mean"),
    ).round(2)
    aggregated.to_csv(REPORTS_DIR / "cluster_profile_summary.csv", index=False)

    profile_lines = ["# Interpretasi Hasil Clustering", ""]
    for _, row in aggregated.iterrows():
        profile_lines.extend(
            [
                f"## Cluster {int(row['Target'])}",
                f"- Jumlah transaksi: {int(row['transaction_count'])}",
                f"- Rata-rata nominal transaksi: {row['avg_transaction_amount']}",
                f"- Rata-rata spending deviation score: {row['avg_spending_deviation']}",
                f"- Rata-rata velocity score: {row['avg_velocity_score']}",
                f"- Rata-rata geo anomaly score: {row['avg_geo_anomaly_score']}",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(profile_lines), encoding="utf-8")
    return labeled_training


def build_decision_tree(training_df: pd.DataFrame) -> dict[str, float | str]:
    X = training_df.drop(columns=["Target"])
    y = training_df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODELS_DIR / "decision_tree_model.h5")
    joblib.dump(model, SUBMISSION_CLASSIFICATION_MODEL)
    (REPORTS_DIR / "classification_report.txt").write_text(classification_report(y_test, predictions), encoding="utf-8")

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy),
    }
    (REPORTS_DIR / "classification_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def plot_cluster_distribution(training_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.countplot(data=training_df, x="Target", hue="Target", palette="Set2", legend=False)
    plt.title("Distribusi Hasil Cluster")
    plt.xlabel("Target Cluster")
    plt.ylabel("Jumlah Data")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_distribution.png", dpi=180)
    plt.close()


def assemble_submission_artifacts() -> dict[str, str]:
    notebook_sources = {
        SOURCE_CLUSTER_NOTEBOOK: SUBMISSION_CLUSTER_NOTEBOOK,
        SOURCE_CLASSIFICATION_NOTEBOOK: SUBMISSION_CLASSIFICATION_NOTEBOOK,
    }
    copied_files: dict[str, str] = {}

    for source_path, destination_path in notebook_sources.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Notebook submission tidak ditemukan: {source_path}")
        shutil.copy2(source_path, destination_path)
        copied_files[destination_path.name] = str(destination_path)

    required_artifacts = [
        SUBMISSION_CLUSTER_NOTEBOOK,
        SUBMISSION_CLASSIFICATION_NOTEBOOK,
        SUBMISSION_CLUSTER_MODEL,
        SUBMISSION_CLASSIFICATION_MODEL,
        SUBMISSION_CLUSTER_DATA,
    ]
    missing_artifacts = [str(path) for path in required_artifacts if not path.exists()]
    if missing_artifacts:
        raise FileNotFoundError(f"Artefak submission belum lengkap: {missing_artifacts}")

    copied_files["model_clustering.h5"] = str(SUBMISSION_CLUSTER_MODEL)
    copied_files["decision_tree_model.h5"] = str(SUBMISSION_CLASSIFICATION_MODEL)
    copied_files["data_clustering.csv"] = str(SUBMISSION_CLUSTER_DATA)
    return copied_files


def save_project_summary(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    best_k: int,
    metrics_df: pd.DataFrame,
    classification_metrics: dict[str, float | str],
    submission_artifacts: dict[str, str],
) -> None:
    summary = {
        "dataset_source": KAGGLE_DATASET_SLUG,
        "sample_size": SAMPLE_SIZE,
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(cleaned_df)),
        "best_cluster_count": best_k,
        "cluster_metrics": metrics_df.to_dict(orient="records"),
        "classification_metrics": classification_metrics,
        "artifacts": {
            "raw_dataset": str(RAW_DATASET_PATH),
            "preprocessed_dataset": str(PREPROCESSED_DATASET_PATH),
            "training_with_target": str(CLUSTERED_DATASET_PATH),
            "clustering_model": str(MODELS_DIR / "model_clustering.joblib"),
            "clustering_model_h5": str(MODELS_DIR / "model_clustering.h5"),
            "decision_tree_model": str(MODELS_DIR / "decision_tree_model.h5"),
            "submission_folder": str(SUBMISSION_DIR),
            "submission_artifacts": submission_artifacts,
        },
    }
    PROJECT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_pipeline() -> dict[str, object]:
    ensure_directories()
    clean_submission_directory()
    stage_kaggle_dataset()
    raw_df = load_dataset()
    cleaned_df, encoded_df, _, _, _ = preprocess_dataset(raw_df)
    best_k, metrics_df = find_best_cluster(encoded_df)
    _, labels = build_clustering_model(encoded_df, best_k)
    training_df = save_cluster_outputs(cleaned_df, encoded_df, labels)
    classification_metrics = build_decision_tree(training_df)
    plot_cluster_distribution(training_df)
    submission_artifacts = assemble_submission_artifacts()
    save_project_summary(raw_df, cleaned_df, best_k, metrics_df, classification_metrics, submission_artifacts)

    print("Pipeline submission machine learning selesai dijalankan.")
    print(f"Sumber dataset: {KAGGLE_DATASET_SLUG}")
    print(f"Dataset mentah tersimpan di: {RAW_DATASET_PATH}")
    print(f"Jumlah cluster terbaik: {best_k}")
    print(f"Akurasi Decision Tree: {classification_metrics['accuracy']:.4f}")

    return {
        "raw_dataset": str(RAW_DATASET_PATH),
        "clean_rows": len(cleaned_df),
        "best_cluster_count": best_k,
        "classification_accuracy": classification_metrics["accuracy"],
    }
