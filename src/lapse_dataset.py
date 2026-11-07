from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Load cleaned data
orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")
customers = pd.read_csv(ROOT / "data" / "clean" / "customers.csv")

# Convert dates
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Define cutoff date
cutoff = pd.Timestamp("2025-09-01")

# Use completed orders only
completed = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < cutoff)
].copy()

# Customers who ordered before cutoff
eligible_ids = completed["customer_id"].unique()

# Create lapse labels
later_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= cutoff) &
    (orders["order_ts"] <= pd.Timestamp("2025-09-30 23:59:59"))
]

later_ids = set(later_orders["customer_id"])

dataset = pd.DataFrame({"customer_id": eligible_ids})

dataset["lapsed"] = (
    ~dataset["customer_id"].isin(later_ids)
).astype(int)

print("Eligible customers:", len(dataset))
print("Lapsed customers:", dataset["lapsed"].sum())
print("Not lapsed:", (dataset["lapsed"] == 0).sum())
print("Lapse percentage:", round(dataset["lapsed"].mean() * 100, 2))

# Save dataset
output = ROOT / "outputs" / "lapse_dataset.csv"
dataset.to_csv(output, index=False)

print("\nLapse dataset saved successfully!")