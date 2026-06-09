import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="DBSCAN Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.hero{
    padding:25px;
    border-radius:15px;
    background:linear-gradient(135deg,#059669,#0284c7);
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.metric-card{
    padding:15px;
    border-radius:12px;
    background:white;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HERO
# ------------------------------------------------

st.markdown("""
<div class='hero'>
<h1>🔍 DBSCAN Clustering Dashboard</h1>
<p>Density Based Clustering using Iris Dataset</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ------------------------------------------------
# LOAD SAVED MODEL
# ------------------------------------------------

# ------------------------------------------------
# TRAIN DBSCAN MODEL
# ------------------------------------------------

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Train DBSCAN
model = DBSCAN(
    eps=0.8,
    min_samples=5
)

labels = model.fit_predict(scaled_data)

df["Cluster"] = labels

# ------------------------------------------------
# METRICS
# ------------------------------------------------

clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_points = list(labels).count(-1)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Samples", len(df))

with c2:
    st.metric("Clusters Found", clusters)

with c3:
    st.metric("Noise Points", noise_points)

# ------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(df.head(15), use_container_width=True)

# ------------------------------------------------
# CLUSTER DISTRIBUTION
# ------------------------------------------------

st.subheader("📊 Cluster Distribution")

cluster_counts = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)

fig1, ax1 = plt.subplots(figsize=(7,4))

ax1.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

ax1.set_xlabel("Cluster")
ax1.set_ylabel("Count")
ax1.set_title("Cluster Distribution")

st.pyplot(fig1)

# ------------------------------------------------
# FEATURE SELECTION
# ------------------------------------------------

st.subheader("🎯 Cluster Visualization")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "Select X Feature",
        iris.feature_names,
        index=0
    )

with col2:
    y_feature = st.selectbox(
        "Select Y Feature",
        iris.feature_names,
        index=2
    )

fig2, ax2 = plt.subplots(figsize=(8,5))

scatter = ax2.scatter(
    df[x_feature],
    df[y_feature],
    c=labels
)

ax2.set_xlabel(x_feature)
ax2.set_ylabel(y_feature)
ax2.set_title("DBSCAN Cluster Visualization")

st.pyplot(fig2)

# ------------------------------------------------
# NOISE POINTS
# ------------------------------------------------

st.subheader("🚨 Noise Points")

noise_df = df[df["Cluster"] == -1]

if len(noise_df) > 0:
    st.dataframe(noise_df, use_container_width=True)
else:
    st.success("No noise points detected.")

# ------------------------------------------------
# CLUSTER STATISTICS
# ------------------------------------------------

st.subheader("📑 Cluster Statistics")

cluster_stats = (
    df.groupby("Cluster")
    .mean()
)

st.dataframe(
    cluster_stats,
    use_container_width=True
)

# ------------------------------------------------
# DOWNLOAD RESULTS
# ------------------------------------------------

csv = df.to_csv(index=False)

st.download_button(
    label="⬇ Download Clustered Dataset",
    data=csv,
    file_name="dbscan_results.csv",
    mime="text/csv"
)

# ------------------------------------------------
# THEORY SECTION
# ------------------------------------------------


