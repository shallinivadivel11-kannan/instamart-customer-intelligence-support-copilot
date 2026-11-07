
from pathlib import Path
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"

# Load cleaned data
orders = pd.read_csv(CLEAN / "orders.csv")
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")

# Convert numeric columns
orders["order_amount"] = pd.to_numeric(orders["order_amount"])
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"])
orders["quantity"] = pd.to_numeric(orders["quantity"])
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"])

# Keep completed orders only
completed = orders[orders["status"] == "completed"].copy()

# Merge customer and product information
merged = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
).merge(
    products[["product_id", "category"]],
    on="product_id"
)

# ------------------------------------------------
# 1. PANDAS PIVOT TABLE - CROSS-CHECK SQL Q2
# ------------------------------------------------
print("\n========== PANDAS PIVOT TABLE ==========")

pivot = pd.pivot_table(
    merged,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print(pivot.round(2))

pivot_total = pivot.loc["Grand Total", "Grand Total"]

# Compare with SQL database total
conn = sqlite3.connect(DB)

sql_total = pd.read_sql_query("""
    SELECT SUM(order_amount) AS total_revenue
    FROM orders
    WHERE status = 'completed'
""", conn)["total_revenue"].iloc[0]

print("\nPandas grand total:", round(pivot_total, 2))
print("SQL grand total:", round(sql_total, 2))

assert abs(pivot_total - sql_total) < 0.01

# ------------------------------------------------
# 2. CITY-WISE SPENDING STATISTICS
# ------------------------------------------------
print("\n========== CITY-WISE ANALYSIS ==========")

city_stats = merged.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_amount=("order_amount", "mean"),
    order_count=("order_amount", "count")
)

print(city_stats.round(2))

# ------------------------------------------------
# 3. ORDER AMOUNT DISTRIBUTION
# ------------------------------------------------
print("\n========== ORDER AMOUNT STATISTICS ==========")

amount = completed["order_amount"]

mean_value = amount.mean()
median_value = amount.median()
skewness = amount.skew()

q1 = amount.quantile(0.25)
q2 = amount.quantile(0.50)
q3 = amount.quantile(0.75)

iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

lower_outliers = (amount < lower_fence).sum()
upper_outliers = (amount > upper_fence).sum()

print("Mean:", round(mean_value, 2))
print("Median:", round(median_value, 2))
print("Skewness:", round(skewness, 2))
print("Q1:", round(q1, 2))
print("Q2:", round(q2, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Outliers below lower fence:", lower_outliers)
print("Outliers above upper fence:", upper_outliers)

# ------------------------------------------------
# 4. CORRELATION MATRIX
# ------------------------------------------------
print("\n========== CORRELATION MATRIX ==========")

columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation = completed[columns].corr()

print(correlation.round(3))

# Find the strongest absolute correlation pair
absolute_corr = correlation.abs()
for col in absolute_corr.columns:
    absolute_corr.loc[col, col] = 0

strongest_pair = absolute_corr.stack().idxmax()

print("\nLargest absolute correlation pair:")
print(strongest_pair)

print(
    "Correlation value:",
    round(correlation.loc[strongest_pair[0], strongest_pair[1]], 3)
)

conn.close()

print("\nPANDAS ANALYSIS COMPLETED SUCCESSFULLY")
