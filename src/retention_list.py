from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent

# Load saved model
model = joblib.load(ROOT / "outputs" / "lapse_model.joblib")

# Load customer features and test split
df = pd.read_csv(ROOT / "outputs" / "lapse_features.csv")
segments = pd.read_csv(ROOT / "outputs" / "customer_segments.csv")

# Recreate the same test split
from sklearn.model_selection import train_test_split

X = df.drop(columns=["customer_id", "lapsed", "signup_date"])
y = df["lapsed"]

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Identify test customer IDs
test_indices = X_test.index
test_ids = df.loc[test_indices, "customer_id"]

# Predict lapse probabilities
probabilities = model.predict_proba(X_test)[:, 1]

predictions = pd.DataFrame({
    "customer_id": test_ids.values,
    "lapse_probability": probabilities,
    "actually_lapsed": y_test.values
})

# Select Premium customers from test set
premium_ids = segments.loc[
    segments["is_premium"] == 1, "customer_id"
]

retention_list = predictions[
    predictions["customer_id"].isin(premium_ids)
].copy()

# Top 10 highest lapse probabilities
retention_list = retention_list.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

# Save output
retention_list.to_csv(
    ROOT / "outputs" / "retention_list.csv",
    index=False
)

print("\nTop 10 Premium Customers for Retention:")
print(retention_list.to_string(index=False))

print("\nActually lapsed customers in retention list:",
      retention_list["actually_lapsed"].sum())

print("\nRetention list saved successfully!")