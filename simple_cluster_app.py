import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
# kmeans clustering
from sklearn.cluster import KMeans
import plotly.express as px

csv_file = st.file_uploader("Upload a CSV file", type=['csv'])

if csv_file is None:
    st.warning("No file uploaded")
    st.stop()

df = pd.read_csv(csv_file)

# drop rows with a missing value
df.dropna(inplace=True)

columns = df.columns.tolist()

label = st.selectbox("Select the column to use as label", columns)

remaining_columns = [col for col in columns if col != label]

data_cols = st.multiselect("Select the columns to use for clustering", remaining_columns, remaining_columns)

kmeans = st.slider("Select the number of clusters", min_value=2, max_value=10, value=2)



n_components = st.slider("Select the number of components for PCA", min_value=2, max_value=3, value=2)

data = df[data_cols].values

kmeans = KMeans(n_clusters=kmeans)
clusters = kmeans.fit_predict(data)
df['cluster'] = clusters

if n_components == 2:
    pcas = PCA(n_components=n_components).fit_transform(data)
    df['pca1'] = pcas[:, 0]
    df['pca2'] = pcas[:, 1]

    fig = px.scatter(df, x='pca1', y='pca2', color='cluster', hover_data=[label] + data_cols)
    st.plotly_chart(fig)

elif n_components == 3:
    pcas = PCA(n_components=n_components).fit_transform(data)
    df['pca1'] = pcas[:, 0]
    df['pca2'] = pcas[:, 1]
    df['pca3'] = pcas[:, 2]

    # 3D plot
    fig = px.scatter_3d(df, x='pca1', y='pca2', z='pca3', color='cluster', hover_data=[label] + data_cols)
    st.plotly_chart(fig)

st.dataframe(df)




