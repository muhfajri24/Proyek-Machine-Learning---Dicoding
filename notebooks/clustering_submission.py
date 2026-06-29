# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from io import StringIO
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from yellowbrick.cluster import KElbowVisualizer

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.transaction_ml_pipeline import (
    CLUSTERED_DATASET_PATH,
    FIGURES_DIR,
    PREPROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    SUBMISSION_CLUSTER_DATA,
    SUBMISSION_CLUSTER_MODEL,
    KAGGLE_DATASET_SLUG,
    stage_kaggle_dataset,
)

# %%
print("Dataset source:", KAGGLE_DATASET_SLUG)
stage_kaggle_dataset()
df = pd.read_csv(RAW_DATASET_PATH)
df.head()

# %%
buffer = StringIO()
df.info(buf=buffer)
print(buffer.getvalue())

# %%
df.describe(include="all").T

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
prepared_df = df.drop(columns=["FraudType", "TimeSinceLastTransaction"], errors="ignore")
cleaned_df = prepared_df.dropna().drop_duplicates().copy()
cleaned_df = cleaned_df.drop(
    columns=[
        "TransactionID",
        "AccountID",
        "CounterpartyAccount",
        "DeviceID",
        "IPAddress",
        "TransactionDate",
        "IsFraud",
    ],
    errors="ignore",
)
cleaned_df.head()

# %%
encoded_df = cleaned_df.copy()
encoders = {}
for column in encoded_df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    encoded_df[column] = encoder.fit_transform(encoded_df[column].astype(str))
    encoders[column] = encoder

encoded_df.to_csv(PREPROCESSED_DATASET_PATH, index=False)
encoded_df.head()

# %%
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(encoded_df), columns=encoded_df.columns)
scaled_df.head()

# %%
elbow_model = KMeans(random_state=42, n_init=20)
elbow_model._estimator_type = "clusterer"
visualizer = KElbowVisualizer(elbow_model, k=(2, 8), timings=False)
visualizer.fit(scaled_df)
visualizer.show(outpath=str(FIGURES_DIR / "notebook_elbow_method.png"))
best_k = visualizer.elbow_value_
if best_k is None:
    best_k = 2
best_k

# %%
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(scaled_df)
joblib.dump(kmeans, SUBMISSION_CLUSTER_MODEL)

# %%
clustered_df = encoded_df.copy()
clustered_df["Target"] = cluster_labels
clustered_df.to_csv(CLUSTERED_DATASET_PATH, index=False)
clustered_df.to_csv(SUBMISSION_CLUSTER_DATA, index=False)
clustered_df.head()

# %%
cleaned_with_target = cleaned_df.copy()
cleaned_with_target["Target"] = cluster_labels
numeric_columns = cleaned_with_target.select_dtypes(include="number").columns.drop("Target")
cluster_summary = cleaned_with_target.groupby("Target")[numeric_columns].agg(["mean", "min", "max"])
cluster_summary

# %%
cleaned_with_target.groupby("Target").agg(
    transaction_count=("TransactionAmount", "count"),
    avg_transaction_amount=("TransactionAmount", "mean"),
    avg_spending_deviation=("SpendingDeviationScore", "mean"),
    avg_velocity_score=("VelocityScore", "mean"),
    avg_geo_anomaly_score=("GeoAnomalyScore", "mean"),
)
