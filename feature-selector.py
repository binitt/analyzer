import pandas as pd

input_csv = "data/software_defect_prediction_dataset.csv"  # full dataset
output_csv = "data/train_filtered.csv"                     # filtered dataset


selected_columns = [
    "lines_of_code",
    "cyclomatic_complexity",
    "num_functions",
    "num_classes",
    "comment_density",
    "code_churn",
    "num_developers",
    "commit_frequency",
    "avg_function_length",
    "bug_fix_commits",

    "past_defects",
#    "developer_experience_years",
#    "test_coverage",
#    "duplication_percentage",
#    "depth_of_inheritance",
#    "response_for_class",
#    "coupling_between_objects",
#    "lack_of_cohesion",
#    "build_failures",
#    "static_analysis_warnings",
#    "security_vulnerabilities",
#    "performance_issues",


    "defect"
]

df = pd.read_csv(input_csv)


# keep only columns we can compute
existing_cols = [c for c in selected_columns if c in df.columns]

df_filtered = df[existing_cols]


df_filtered.to_csv(output_csv, index=False)


print("Original dataset shape:", df.shape)
print("Filtered dataset shape:", df_filtered.shape)
print("Columns used:", existing_cols)
print("Training dataset saved:", output_csv)
