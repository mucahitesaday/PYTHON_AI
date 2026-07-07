import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('output', exist_ok=True)

print("=" * 60)
print("ONLINE RETAIL RFM SEGMENTATION")
print("=" * 60)

print("\n[1] Loading data...")
df = pd.read_excel('online_retail.xlsx')
print(f"    Shape: {df.shape}")

print("\n[2] Cleaning data...")
print(f"    Missing CustomerID: {df['CustomerID'].isnull().sum()}")
df = df.dropna(subset=['CustomerID'])
print(f"    After dropna CustomerID: {df.shape}")

df = df[df['Quantity'] > 0]
print(f"    After removing negative/zero Quantity: {df.shape}")

df = df[df['UnitPrice'] > 0]
print(f"    After removing negative/zero UnitPrice: {df.shape}")

df = df[~df['InvoiceNo'].astype(str).str.contains('C', na=False)]
print(f"    After removing cancellations: {df.shape}")

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("\n[3] Creating RFM features...")

reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

print(f"    RFM shape: {rfm.shape}")
print(f"    Recency range: {rfm['Recency'].min()} - {rfm['Recency'].max()} days")
print(f"    Frequency range: {rfm['Frequency'].min()} - {rfm['Frequency'].max()}")
print(f"    Monetary range: {rfm['Monetary'].min():.2f} - {rfm['Monetary'].max():.2f}")

print("\n[4] Scaling features...")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

print("\n[5] Finding optimal clusters (Elbow method)...")
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(rfm_scaled, labels)
    silhouettes.append(sil)
    print(f"    K={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')

axes[1].plot(K_range, silhouettes, 'ro-')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.savefig('output/optimal_k.png', dpi=150)
print("    Saved: output/optimal_k.png")

best_k = K_range[np.argmax(silhouettes)]
print(f"\n    Best K (by Silhouette): {best_k}")

print(f"\n[6] Running K-Means with K={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("\n[7] Cluster profiles:")
cluster_profile = rfm.groupby('Cluster').agg(
    Count=('CustomerID', 'count'),
    Recency_mean=('Recency', 'mean'),
    Frequency_mean=('Frequency', 'mean'),
    Monetary_mean=('Monetary', 'mean')
).round(2)

cluster_profile['Pct'] = (cluster_profile['Count'] / cluster_profile['Count'].sum() * 100).round(1)
print(cluster_profile.to_string())

print("\n[8] Saving results...")
rfm.to_csv('output/rfm_segments.csv', index=False)
print("    Saved: output/rfm_segments.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=rfm, x='Cluster', y='Recency', ax=axes[0])
axes[0].set_title('Recency by Cluster')

sns.boxplot(data=rfm, x='Cluster', y='Frequency', ax=axes[1])
axes[1].set_title('Frequency by Cluster')

sns.boxplot(data=rfm, x='Cluster', y='Monetary', ax=axes[2])
axes[2].set_title('Monetary by Cluster')

plt.tight_layout()
plt.savefig('output/cluster_profiles.png', dpi=150)
print("    Saved: output/cluster_profiles.png")

print("\n[9] Segment labels:")
segment_names = {}
for c in sorted(rfm['Cluster'].unique()):
    row = cluster_profile.loc[c]
    if row.Recency_mean < cluster_profile.Recency_mean.median() and row.Monetary_mean > cluster_profile.Monetary_mean.median():
        label = "VIP / High-Value (Sadık)"
    elif row.Recency_mean < cluster_profile.Recency_mean.median() and row.Frequency_mean > cluster_profile.Frequency_mean.median():
        label = "Aktif Düzenli Müşteri"
    elif row.Recency_mean > cluster_profile.Recency_mean.median() and row.Monetary_mean > cluster_profile.Monetary_mean.median():
        label = "Eski Ama Yüksek Harcama"
    elif row.Recency_mean > cluster_profile.Recency_mean.median() and row.Monetary_mean < cluster_profile.Monetary_mean.median():
        label = "Kaybetme Riski (Düşük Aktivite)"
    else:
        label = "Potansiyel Gelişim"
    segment_names[c] = label
    print(f"    Cluster {c}: {label} ({row['Count']} müşteri, {row['Pct']}%)")

rfm['Segment'] = rfm['Cluster'].map(segment_names)
rfm.to_csv('output/rfm_segments_labeled.csv', index=False)
print("    Saved: output/rfm_segments_labeled.csv")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
