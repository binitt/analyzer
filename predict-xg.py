import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "defect_model_xgboost.pkl")
DATA_TO_PREDICT = os.path.join(PROJECT_ROOT, "data", "project-dataset.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "output", "predict-xg.csv")

def run_prediction():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(DATA_TO_PREDICT):
        print(f"❌ Data file not found: {DATA_TO_PREDICT}")
        return

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_TO_PREDICT)

    if df.empty:
        print("❌ CSV file is empty!")
        return

    print(df.head())

    if "module" not in df.columns:
        print("❌ 'module' column missing!")
        return

    module_names = df["module"]

    y_true = None
    if "defect" in df.columns:
        y_true = df["defect"]
        df = df.drop("defect", axis=1)

    df = df.drop("module", axis=1)
    numeric_df = df.select_dtypes(include=['number'])

    expected_features = [
        'lines_of_code', 'cyclomatic_complexity', 'num_functions',
        'num_classes', 'comment_density', 'code_churn',
        'num_developers', 'commit_frequency',
        'avg_function_length', 'bug_fix_commits', 'past_defects'
    ]

    for col in expected_features:
        if col not in numeric_df.columns:
            numeric_df[col] = 0

    numeric_df = numeric_df[expected_features]

    predictions = model.predict(numeric_df)
    probabilities = model.predict_proba(numeric_df)[:, 1]

    output_df = pd.DataFrame({
        "module": module_names,
        "defect": predictions
    })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False)

    if y_true is not None:
        try:
            acc = accuracy_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)
            roc = roc_auc_score(y_true, probabilities)

            print("\n📊 Evaluation Metrics")
            print("="*30)
            print(f"Accuracy : {acc:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print(f"ROC-AUC  : {roc:.4f}")
        except Exception as e:
            print("⚠️ Could not compute metrics:", e)

    print("\n" + "="*30)
    print(f"Total files   : {len(numeric_df)}")
    print(f"Defects found : {(predictions == 1).sum()}")
    print(f"Saved to      : {OUTPUT_FILE}")
    print("="*30 + "\n")

if __name__ == "__main__":
    run_prediction()