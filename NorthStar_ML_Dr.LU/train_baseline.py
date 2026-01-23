import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

OUT = "data/injections"  # <- change
df = pd.read_csv(f"{OUT}/dataset_stage1.csv")

y = df["n_star"].astype(str)
X = df.drop(columns=["n_star"])
X = X.select_dtypes(include=[np.number]).fillna(0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nReport:\n", classification_report(y_test, pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))

# Save model + feature list
joblib.dump({"model": model, "features": list(X.columns)}, f"{OUT}/rf_nstar.joblib")
print("\nSaved:", f"{OUT}/rf_nstar.joblib")

# Print top feature importances
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 feature importances:\n", imp.head(15))
