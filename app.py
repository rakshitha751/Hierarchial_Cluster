import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# ------------------ Streamlit Title ------------------
st.title("🛒 Customer Segmentation using Hierarchical Clustering")

# ------------------ Load Dataset ------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "/home/intellact/Downloads/HIERARCHICAL_CLUSTERING/OnlineRetail.csv",
        encoding="latin1"
    )
    return df

df = load_data()
st.write("Raw Dataset Shape:", df.shape)

# ------------------ Data Cleaning ------------------
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# ------------------ Customer-Level Aggregation ------------------
customer_df = df.groupby("CustomerID").agg({
    "Quantity": "sum",
    "UnitPrice": "mean"
})

st.write("Customer-level Data Shape:", customer_df.shape)
st.dataframe(customer_df.head())

# ------------------ Standardization ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_df)

# ------------------ Sampling for Hierarchical Clustering ------------------
sample_size = st.slider(
    "Select number of customers for dendrogram (Max 2000)",
    min_value=200,
    max_value=2000,
    value=500,
    step=100
)

X_sample = X_scaled[:sample_size]

# ------------------ Hierarchical Clustering ------------------
Z = linkage(X_sample, method="ward")

# ------------------ Dendrogram Plot ------------------
st.subheader("📊 Hierarchical Clustering Dendrogram")

fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=30)
plt.xlabel("Cluster Size")
plt.ylabel("Distance")

st.pyplot(fig)
