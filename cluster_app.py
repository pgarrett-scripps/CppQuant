import pandas as pd
import streamlit as st


with st.sidebar:
    protein_compare = st.file_uploader("Choose protein compare file", type="csv")

    kmeans = st.number_input("Number of clusters", min_value=1, value=3)
    num_significant = st.number_input("Number of significant peptides", min_value=1, value=3)
    significant_threshold = st.number_input("Significant threshold", min_value=0.0, value=0.05)


if protein_compare is None:
    st.stop()


df = pd.read_csv(protein_compare)


# Define a function to filter groups based on qvalue threshold
def filter_proteins(dataframe, threshold):
    # Group by 'protein_site_str' and filter
    grouped = dataframe.groupby('protein_site_str')
    filtered_groups = []

    for name, group in grouped:
        # Check if the number of significant qvalues within the group meets the threshold
        if (group['qvalue'] < threshold).sum() >= num_significant:
            filtered_groups.append(group)

    # Concatenate all qualifying groups back into a single DataFrame
    if filtered_groups:
        return pd.concat(filtered_groups)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no groups qualify


# Apply the filtering function
filtered_df = filter_proteins(df, significant_threshold)

# for each group of protein_site_str, sort by group1, group2, and create an array with all diff std
# Process each group, sort by 'group1' and 'group2', and calculate the std of differences
data = []
groups = []
data_cols = None
for _, group in filtered_df.groupby('protein_site_str'):

    if data_cols is None:
        data_cols = group[['group1', 'group2']].values

    # make an array with the log2_ratio_diff values
    groups.append(_)
    data.append(group['log2_ratio_diff'].values)

# fix data_cols format
data_cols = [f"{x[0]}-{x[1]}" for x in data_cols]

# Create a DataFrame with the std of differences for each group
std_diff_df = pd.DataFrame(data, index=groups, columns=data_cols)


# replace nan with 0
std_diff_df.fillna(0, inplace=True)

# kmeans clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=kmeans)
clusters = kmeans.fit_predict(std_diff_df)


# use dim reduction to plot the clusters
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
std_diff_df[['pca1', 'pca2']] = pca.fit_transform(std_diff_df)

std_diff_df['cluster'] = clusters


# rename index to protein_site_str
std_diff_df.reset_index(inplace=True)


# Display the cluster assignments plotly

import plotly.express as px

fig = px.scatter(std_diff_df, x='pca1', y='pca2', color='cluster', hover_data=['cluster', 'index'])
st.plotly_chart(fig)

st.dataframe(std_diff_df)


# Display the filtered DataFrame
if not filtered_df.empty:
    st.write("Filtered Data:", filtered_df)
else:
    st.write("No data meets the criteria.")
