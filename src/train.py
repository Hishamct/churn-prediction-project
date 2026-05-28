import pandas as pd
import joblib
# import subprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("data/data.csv")

# ---------------------------
# Clean Data
# ---------------------------
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

df = df.drop("customerID", axis=1)

# ---------------------------
# Encode Target
# ---------------------------
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# ---------------------------
# Encode Categorical Features
# ---------------------------
categorical_cols = df.select_dtypes(
    include=["object"]
).columns

encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ---------------------------
# Split Data
# ---------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

# ---------------------------
# MLflow Tracking
# ---------------------------
# mlflow.set_experiment(
#     "churn_prediction"
# )

# with mlflow.start_run():

#     mlflow.log_metric(
#         "accuracy",
#         accuracy
#     )

#     mlflow.sklearn.log_model(
#         model,
#         "model"
#     )

#     commit = subprocess.getoutput(
#         "git rev-parse HEAD"
#     )

#     mlflow.log_param(
#         "git_commit",
#         commit
#     )

# ---------------------------
# Save Model
# ---------------------------
joblib.dump(
    model,
    "model.pkl"
)

print(
    f"Accuracy: {accuracy}"
)