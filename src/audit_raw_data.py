
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# Load the three raw datasets
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

datasets = {
    "CUSTOMERS": customers,
    "PRODUCTS": products,
    "ORDERS": orders
}

# 1. Shape, data types and missing values
for name, df in datasets.items():
    print(f"\n========== {name} ==========")
    print("Shape:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Missing_Percentage": (df.isna().mean() * 100).round(2)
    })
    print(missing)

# 2. Specific audit figures
print("\n========== RAW DATA AUDIT CHECKPOINTS ==========")

numeric_amount = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

numeric_delivery = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

numeric_age = pd.to_numeric(
    customers["age"], errors="coerce"
)

print("Raw order rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending orders:", (orders["status"].str.strip().str.lower() == "pending").sum())
print("Distinct order IDs:", orders["order_id"].nunique())

print(
    "Non-numeric order amounts:",
    numeric_amount.isna().sum()
)

print("Negative delivery times:", (numeric_delivery < 0).sum())
print(
    "Blank delivery times:",
    orders["delivery_minutes"].isna().sum()
    + orders["delivery_minutes"].astype(str).str.strip().eq("").sum()
)

print(
    "Blank customer ages:",
    customers["age"].isna().sum()
    + customers["age"].astype(str).str.strip().eq("").sum()
)

print(
    "Ages outside 10-100:",
    ((numeric_age < 10) | (numeric_age > 100)).sum()
)

print("Distinct raw city spellings:", customers["city"].nunique())

nonblank_gender = customers["gender"].dropna().astype(str).str.strip()
nonblank_gender = nonblank_gender[nonblank_gender != ""]

print("Distinct non-blank gender spellings:", nonblank_gender.nunique())

print(
    "Blank genders:",
    customers["gender"].isna().sum()
    + customers["gender"].astype(str).str.strip().eq("").sum()
)
