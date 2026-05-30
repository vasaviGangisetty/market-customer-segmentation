import os
print("Current Directory:", os.getcwd())

from src.data_preprocessing import load_data, preprocess_data
from src.rfm_analysis import create_rfm_features
from src.pca_analysis import apply_pca
from src.clustering import find_elbow, apply_kmeans
from src.visualization import (
    plot_elbow,
    plot_clusters,
    plot_pca_clusters
)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load Dataset
print("Loading dataset...")
df = load_data("dataset/Mall_Customers.csv")

# Preprocess Data
print("Preprocessing data...")
df, scaled_features = preprocess_data(df)

# RFM Analysis
print("Creating RFM features...")
rfm = create_rfm_features(df)

print("\n")
print("=" * 70)
print("RFM ANALYSIS")
print("=" * 70)
print(rfm)

# PCA
print("\nApplying PCA...")
pca_data = apply_pca(scaled_features)

# Elbow Method
print("\nFinding optimal clusters...")
inertias = find_elbow(scaled_features)

print("Saving elbow graph...")
plot_elbow(inertias)

# KMeans Clustering
print("Applying KMeans...")
labels, model = apply_kmeans(
    scaled_features,
    n_clusters=3
)

# Add Cluster Column
df["Cluster"] = labels

# Save CSV Report
df.to_csv(
    "outputs/cluster_report.csv",
    index=False
)

# Visualizations
print("Generating visualizations...")
plot_clusters(scaled_features, labels)
plot_pca_clusters(pca_data, labels)

# CUSTOMER SEGMENTATION
print("\n")
print("=" * 70)
print("CUSTOMER SEGMENTATION")
print("=" * 70)

print(df[[
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Cluster"
]])

# CUSTOMER SUMMARY
print("\n")
print("=" * 70)
print("CUSTOMER SUMMARY")
print("=" * 70)

summary = df.groupby("Cluster").agg({
    "CustomerID": "count",
    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean"
}).round(2)

summary.columns = [
    "Total Customers",
    "Average Age",
    "Average Income",
    "Average Spending Score"
]

print(summary)

print("\nProject Executed Successfully!")
print("Results saved in outputs folder")