import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
import joblib


TRAIN_DATA = "data/train_filtered.csv"
#TRAIN_DATA = "data/software_defect_prediction_dataset.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


print("Loading dataset:", TRAIN_DATA)
df = pd.read_csv(TRAIN_DATA)

print("Dataset shape:", df.shape)

X = df.drop("defect", axis=1)
y = df["defect"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


pos = (y_train == 1).sum()
neg = (y_train == 0).sum()

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Class distribution:", {0: neg, 1: pos})


models = {

    "naive_bayes":
        GaussianNB(),

    "logistic_regression":
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        ),

    "random_forest":
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    "xgboost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
}


for name, model in models.items():

    print("\n==============================")
    print("Training model:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("Accuracy:", round(acc, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    MODEL_FILE = f"{MODEL_DIR}/defect_model_{name}.pkl"
    joblib.dump(model, MODEL_FILE)

    print("Saved model:", MODEL_FILE)


print("\nTraining complete.")
