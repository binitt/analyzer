import pandas as pd
import joblib

# Files
MODEL_FILE = "models/defect_model_naive_bayes.pkl"
DATA_FILE = "data/dataset.csv"
OUTPUT_FILE = "data/predictions_nb.csv"

# Load the trained Naive Bayes model
model = joblib.load(MODEL_FILE)

# Load the dataset to predict on
df = pd.read_csv(DATA_FILE)

# Features used for prediction (same as training)
features = [
    'lines_of_code', 'cyclomatic_complexity', 'num_functions', 'num_classes',
    'comment_density', 'code_churn', 'num_developers', 'commit_frequency',
    'avg_function_length', 'bug_fix_commits', 'past_defects'
]

# Select features
X = df[features]

# Make predictions
predictions = model.predict(X)

# Add predictions to the dataframe
df['predicted_defect'] = predictions

# Save the results
df.to_csv(OUTPUT_FILE, index=False)

print("Predictions completed and saved to:", OUTPUT_FILE)
print("Sample predictions:")
print(df[['file_name', 'predicted_defect']].head())