import pandas as pd

input_csv = "data/software_defect_prediction_dataset.csv" # full
output_csv = "data/train_filtered.csv" # filtered

selected_columns = [
    "lines_of_code",
    "cyclomatic_complexity",
    "code_churn",
    "num_developers",
    "commit_frequency",
    "defect"
]

df = pd.read_csv(input_csv)

existing_cols = [c for c in selected_columns if c in df.columns]

df_filtered = df[existing_cols]

df_filtered.to_csv(output_csv, index=False)

print("Training dataset saved:", output_csv)
