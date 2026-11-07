
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "instamart.db"
SQL_FILE = ROOT / "sql" / "queries.sql"

# Read all seven SQL queries
sql_text = SQL_FILE.read_text(encoding="utf-8")
queries = [
    q.strip()
    for q in sql_text.split(";")
    if q.strip()
]

conn = sqlite3.connect(DB)

print("Total SQL queries:", len(queries))

results = {}

# Execute each query using pandas
for i, query in enumerate(queries, start=1):
    df = pd.read_sql_query(query, conn)
    results[i] = df

    print(f"\n========== QUERY Q{i} ==========")
    print("Rows returned:", len(df))
    print(df.head(5).to_string(index=False))

# Q1 checkpoint
assert len(results[1]) == 30

# Q2 checkpoint
assert len(results[2]) == 30
assert abs(results[2]["revenue"].sum() - 2998668.80) < 0.01

# Largest Q2 revenue cell
largest = results[2].loc[results[2]["revenue"].idxmax()]
print("\nQ2 largest revenue cell:")
print(largest.to_dict())

# Q3 checkpoint
assert len(results[3]) == 11

# Q4 checkpoint
assert len(results[4]) == 25

# Q5 checkpoint
assert len(results[5]) == 550

# Q5 average customer spending
avg_spend = pd.read_sql_query("""
    SELECT AVG(total_spend) AS avg_spend
    FROM (
        SELECT customer_id, SUM(order_amount) AS total_spend
        FROM orders
        WHERE status = 'completed'
        GROUP BY customer_id
    )
""", conn)["avg_spend"].iloc[0]

print("\nQ5 average customer spending:", round(avg_spend, 2))

# Q6 ranking tie check
q6 = results[6]
tie_count = (q6["spending_rank"] != q6["row_number"]).sum()
print("\nQ6 rows where RANK differs from ROW_NUMBER:", tie_count)

# Q6 top customer in each city
print("\nQ6 top customer per city:")
print(
    q6.sort_values(
        ["city", "total_spend"],
        ascending=[True, False]
    ).groupby("city").head(1).to_string(index=False)
)

# Q7 checkpoint
assert len(results[7]) == 183
print("\nQ7 final cumulative revenue:")
print(results[7].iloc[-1].to_dict())

conn.close()

print("\nALL SEVEN SQL QUERIES EXECUTED SUCCESSFULLY")
