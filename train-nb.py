import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# files
TRAIN_DATA = "data/train_filtered.csv"
MODEL_FILE = "models/defect_model_nb.pkl"

# load dataset
df = pd.read_csv(TRAIN_DATA)

# features and target
X = df.drop(columns=["defect"])
y = df["defect"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = GaussianNB()
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# persist model
joblib.dump(model, MODEL_FILE)

print("Model saved to:", MODEL_FILE)
