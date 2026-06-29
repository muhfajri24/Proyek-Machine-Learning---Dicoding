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
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.transaction_ml_pipeline import (
    CLUSTERED_DATASET_PATH,
    REPORTS_DIR,
    SUBMISSION_CLASSIFICATION_MODEL,
    SUBMISSION_CLUSTER_DATA,
)

# %%
training_df = pd.read_csv(SUBMISSION_CLUSTER_DATA if SUBMISSION_CLUSTER_DATA.exists() else CLUSTERED_DATASET_PATH)
training_df.head()

# %%
X = training_df.drop(columns=["Target"])
y = training_df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
decision_tree = DecisionTreeClassifier(max_depth=6, random_state=42)
decision_tree.fit(X_train, y_train)
predictions = decision_tree.predict(X_test)
accuracy_score(y_test, predictions)

# %%
print(classification_report(y_test, predictions))

# %%
joblib.dump(decision_tree, SUBMISSION_CLASSIFICATION_MODEL)
joblib.dump(decision_tree, PROJECT_ROOT / "models" / "decision_tree_model.h5")
