
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

CLEAN.mkdir(parents=True, exist_ok=True)

# Load raw customer and product data
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")

# 1. Standardise city names
customers["city"] = customers["city"].str.strip().str.title()

# 2. Standardise gender values
def clean_gender(value):
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"

    value = str(value).strip().lower()

    if value in ["m", "male", "man"]:
        return "male"
    elif value in ["f", "female", "woman"]:
        return "female"
    else:
        return "unknown"

customers["gender"] = customers["gender"].apply(clean_gender)

# 3. Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# 4. Treat ages outside 10-100 as missing
customers.loc[
    (customers["age"] < 10) | (customers["age"] > 100),
    "age"
] = pd.NA

# 5. Fill missing ages using city median
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

# 6. Convert signup date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Save clean customer data
customers.to_csv(CLEAN / "customers.csv", index=False)

# Save clean product data
products.to_csv(CLEAN / "products.csv", index=False)

# 7. Verify customer data
print("Customer shape:", customers.shape)
print("\nCity counts:")
print(customers["city"].value_counts().sort_index())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

print("\nSignup date type:", customers["signup_date"].dtype)

print("\nClean customer and product files saved successfully!")


# ==========================================
# PART A - CLEAN ORDERS
# ==========================================

# 1. Load raw orders
orders = pd.read_csv(RAW / "orders.csv")

print("\n========== ORDER CLEANING ==========")
print("Original order rows:", len(orders))

# 2. Convert order timestamp to datetime
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# 3. Remove exact duplicate rows
before = len(orders)
orders = orders.drop_duplicates()
print("Exact duplicate rows removed:", before - len(orders))

# 4. Keep only the latest record for each order_id
orders = orders.sort_values("order_ts")
before = len(orders)

orders = orders.drop_duplicates(
    subset="order_id", keep="last"
).copy()

print("Older order records removed:", before - len(orders))

# 5. Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# 6. Get product prices and calculate missing amounts
price_map = products.set_index("product_id")["price_inr"]

orders["price_inr"] = orders["product_id"].map(price_map)

missing_amounts = orders["order_amount"].isna().sum()

orders["order_amount"] = orders["order_amount"].fillna(
    (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)
)

print("Missing amounts filled:", missing_amounts)

# 7. Convert delivery time to numeric
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# 8. Set negative delivery times to missing
negative_times = (orders["delivery_minutes"] < 0).sum()

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

print("Negative delivery times corrected:", negative_times)

# 9. Get product category
category_map = products.set_index("product_id")["category"]
orders["category"] = orders["product_id"].map(category_map)

# 10. Fill missing delivery times for completed orders
completed = orders["status"].eq("completed")

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform(lambda x: x.fillna(x.median()))
)

# Cancelled orders must retain missing delivery times
cancelled = orders["status"].eq("cancelled")

orders.loc[cancelled, "delivery_minutes"] = pd.NA

# 11. Remove helper columns
orders = orders.drop(columns=["price_inr", "category"])

# 12. Save clean orders
orders.to_csv(CLEAN / "orders.csv", index=False)

# 13. Verify order-cleaning results
print("\n========== CLEAN ORDER CHECKPOINTS ==========")

print("Final order rows:", len(orders))
print("\nOrder status counts:")
print(orders["status"].value_counts())

print(
    "\nMissing order amounts:",
    orders["order_amount"].isna().sum()
)

print(
    "Missing delivery times in completed orders:",
    orders.loc[completed, "delivery_minutes"].isna().sum()
)

print(
    "Missing delivery times in cancelled orders:",
    orders.loc[cancelled, "delivery_minutes"].isna().sum()
)

print(
    "Completed order revenue (INR):",
    round(orders.loc[completed, "order_amount"].sum(), 2)
)

print("\nClean orders saved successfully!")
