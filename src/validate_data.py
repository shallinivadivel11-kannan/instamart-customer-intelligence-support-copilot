
from pathlib import Path
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"

# Load cleaned CSV files
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Check 1: Customer count
assert len(customers) == 1200

# Check 2: Product count
assert len(products) == 40

# Check 3: Order count
assert len(orders) == 12309

# Check 4: Unique order IDs
assert orders["order_id"].nunique() == 12309

# Check 5: City counts
expected_cities = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184
}
assert customers["city"].value_counts().to_dict() == expected_cities

# Check 6: Gender counts
expected_genders = {
    "female": 610,
    "male": 560,
    "unknown": 30
}
assert customers["gender"].value_counts().to_dict() == expected_genders

# Check 7: No missing customer ages
assert customers["age"].isna().sum() == 0

# Check 8: All customer ages between 10 and 100
assert customers["age"].between(10, 100).all()

# Check 9: No pending orders
assert not orders["status"].str.lower().eq("pending").any()

# Check 10: Order status counts
assert orders["status"].value_counts().to_dict() == {
    "completed": 11110,
    "cancelled": 1199
}

# Check 11: No missing order amounts
assert orders["order_amount"].notna().all()

# Check 12: Completed orders have delivery times
completed = orders["status"].str.lower().eq("completed")
assert orders.loc[completed, "delivery_minutes"].notna().all()

# Check 13: Cancelled orders have missing delivery times
cancelled = orders["status"].str.lower().eq("cancelled")
assert orders.loc[cancelled, "delivery_minutes"].isna().all()

# Check 14: Completed revenue
revenue = orders.loc[completed, "order_amount"].sum()
assert abs(revenue - 2998668.80) < 0.01

# Check 15: SQLite database tables and row counts
conn = sqlite3.connect(DB)

tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn
)["name"].tolist()

assert set(tables) == {"customers", "products", "orders"}

for table, expected_count in {
    "customers": 1200,
    "products": 40,
    "orders": 12309
}.items():
    count = pd.read_sql_query(
        f"SELECT COUNT(*) AS total FROM {table}",
        conn
    )["total"].iloc[0]

    assert count == expected_count

conn.close()

print("ALL CHECKPOINTS PASSED")
