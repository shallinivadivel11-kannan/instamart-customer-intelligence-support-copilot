
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

rfm = pd.read_csv(OUTPUTS / "rfm_table.csv")

features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

# Final K-means model
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(X_scaled)

# Segment profiles
profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

# Name segments using their RFM characteristics
premium_id = profile["mean_monetary"].idxmax()

def name_segment(row):
    if row.name == premium_id:
        return "Premium"
    elif row["mean_frequency"] >= 15:
        return "Loyal"
    elif row["mean_recency"] >= 60:
        return "At Risk"
    else:
        return "Occasional"

name_map = {
    segment_id: name_segment(row)
    for segment_id, row in profile.iterrows()
}

rfm["segment_name"] = rfm["segment_id"].map(name_map)
rfm["is_premium"] = (rfm["segment_id"] == premium_id).astype(int)

# Save customer segments
rfm.to_csv(OUTPUTS / "customer_segments.csv", index=False)

# Save segment profile
profile.to_csv(OUTPUTS / "segment_profile.csv")

# PCA for two-dimensional representation
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

print("\n========== SEGMENT PROFILE ==========")
print(profile)
print("\nSegment names:")
print(name_map)

print("\nPCA explained variance ratios:")
print(pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

print("\nCustomer segments saved successfully!")


print("\nK-means evaluation saved successfully!")
