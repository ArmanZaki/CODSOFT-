
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("Churn_Modelling.csv")

df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = ["CreditScore", "Age", "Tenure", "Balance",
                "NumOfProducts", "EstimatedSalary"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


models = {
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc
    })
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    joblib.dump(model, f"best_model_{name}.joblib")


results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print("\nModel Evaluation Results:\n", results_df)

results_df.to_csv("churn_model_results.csv", index=False)

plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.3f})")
    
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Customer Churn Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.show()

print("\nTraining complete. Models, results CSV, and ROC curves saved.")
