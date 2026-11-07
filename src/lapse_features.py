from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
cutoff = pd.Timestamp("2025-09-01")

orders = pd.read_csv(ROOT / "data/clean/orders.csv")
customers = pd.read_csv(ROOT / "data/clean/customers.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Use only orders before cutoff
history = orders[orders["order_ts"] < cutoff].copy()

completed = history[history["status"] == "completed"].copy()

# Customer-level completed order features
features = completed.groupby("customer_id").agg(
    recency=("order_ts", lambda x: (cutoff - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

# Cancellation rate using all final-status orders before cutoff
status_counts = history.groupby("customer_id")["status"].agg(
    total_orders="count",
    cancelled_orders=lambda x: (x == "cancelled").sum()
).reset_index()

status_counts["cancellation_rate"] = (
    status_counts["cancelled_orders"] / status_counts["total_orders"]
)

# Most frequently ordered product category
category_data = completed.merge(
    pd.read_csv(ROOT / "data/clean/products.csv")[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_category = (
    category_data.groupby("customer_id")["category"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index(name="favourite_category")
)

# Combine features
features = features.merge(status_counts, on="customer_id", how="left")
features = features.merge(favourite_category, on="customer_id", how="left")
features = features.merge(customers, on="customer_id", how="left")

# Tenure as of cutoff
features["signup_date"] = pd.to_datetime(features["signup_date"])
features["tenure_days"] = (
    cutoff - features["signup_date"]
).dt.days

# Add lapse labels
labels = pd.read_csv(ROOT / "outputs/lapse_dataset.csv")
features = features.merge(labels, on="customer_id", how="inner")

# Save
features.to_csv(ROOT / "outputs/lapse_features.csv", index=False)

print("Feature dataset shape:", features.shape)
print("\nMissing values:")
print(features.isna().sum())
print("\nLapse distribution:")
print(features["lapsed"].value_counts())
print("\nLapse features saved successfully!")