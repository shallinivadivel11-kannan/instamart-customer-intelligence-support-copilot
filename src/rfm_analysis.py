
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)

# Load cleaned orders
orders = pd.read_csv(CLEAN / "orders.csv")

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Keep completed orders only
completed = orders[
    orders["status"].str.lower() == "completed"
].copy()

# Snapshot date
snapshot = pd.Timestamp("2025-10-01 00:00:00")

# Calculate RFM
rfm = completed.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency in whole days
rfm["recency"] = (
    snapshot - rfm["last_order_date"]
).dt.days

# Keep required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Save RFM table
rfm.to_csv(OUTPUTS / "rfm_table.csv", index=False)

# Display summary
print("========== RFM ANALYSIS ==========")

print("Total customers:", len(rfm))

print("\nRecency statistics:")
print(rfm["recency"].describe())

print("\nFrequency statistics:")
print(rfm["frequency"].describe())

print("\nMonetary statistics:")
print(rfm["monetary"].describe())

print("\nRFM table saved to outputs/rfm_table.csv")
