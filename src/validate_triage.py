from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "outputs" / "triage_results.csv")

print("========== TRIAGE VALIDATION ==========")

print("Total rows:", len(df))
print("Duplicate ticket IDs:", df["ticket_id"].duplicated().sum())

required_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

print("Correct columns:", list(df.columns) == required_columns)
print("Missing ticket IDs:", df["ticket_id"].isna().sum())

assert len(df) == 164
assert df["ticket_id"].duplicated().sum() == 0
assert list(df.columns) == required_columns
assert df["ticket_id"].notna().all()

print("\nALL TRIAGE CHECKS PASSED")