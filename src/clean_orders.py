
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

# Load raw datasets
orders = pd.read_csv(RAW / "orders.csv")
products = pd.read_csv(RAW / "products.csv")

print("Original order rows:", len(orders))

# 1. Convert order timestamp to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. Remove exact duplicate rows
before = len(orders)
orders = orders.drop_duplicates().copy()
print("Exact duplicate rows removed:", before - len(orders))

# 3. Keep the latest record for each order_id
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset="order_id", keep="last"
).copy()

print("Rows after keeping latest order:", len(orders))

# 4. Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Add product price and category for cleaning
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

# Fill missing amounts using product price and discount
missing_amounts = orders["order_amount"].isna().sum()

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

print("Missing amounts filled:", missing_amounts)

# 5. Convert negative delivery times to missing
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

negative_delivery = (orders["delivery_minutes"] < 0).sum()

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

print("Negative delivery times corrected:", negative_delivery)

# 6. Fill missing delivery times for completed orders
completed = orders["status"].str.strip().str.lower().eq("completed")

category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"]
    .fillna(category_medians)
)

# Cancelled orders must retain missing delivery times

# Remove temporary product columns
orders = orders.drop(columns=["price_inr", "category"])

# Save timestamp in the required text format
orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

# Save cleaned orders
orders.to_csv(CLEAN / "orders.csv", index=False)

# 7. Verify results
print("\n========== CLEAN ORDERS CHECKPOINTS ==========")
print("Clean order rows:", len(orders))
print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nDuplicate order IDs:", orders["order_id"].duplicated().sum())
print("Missing order amounts:", orders["order_amount"].isna().sum())

print(
    "Missing delivery times in completed orders:",
    orders.loc[completed, "delivery_minutes"].isna().sum()
)

print(
    "Missing delivery times in cancelled orders:",
    orders.loc[~completed, "delivery_minutes"].isna().sum()
)

print(
    "Completed order revenue (INR):",
    round(
        orders.loc[completed, "order_amount"].sum(),
        2
    )
)

print("\nClean orders saved successfully!")
