import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# --- Load and clean ---
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# --- Encode and split ---
df = pd.get_dummies(df, drop_first=True)
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Final model: balanced Logistic Regression ---
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))
print("ROC AUC:", round(roc_auc_score(y_test, proba), 3))

# --- Top factors behind churn ---
coefs = pd.Series(
    model.named_steps["logisticregression"].coef_[0], index=X.columns
).sort_values()
print("\nFactors that most reduce churn:")
print(coefs.head(5).round(2))
print("\nFactors that most increase churn:")
print(coefs.tail(5).round(2))

# --- Save the model ---
joblib.dump(model, "churn_model.pkl")
print("\nModel saved as churn_model.pkl")