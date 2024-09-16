import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from minisom import MiniSom
from sklearn.metrics import silhouette_score, silhouette_samples
from datetime import date

# Function to run SOM clustering
def run_som(som_df, som_size, learning_rate, num_iterations):
    # Add Game_Age_Years if it's missing
    if 'Game_Age_Years' not in som_df.columns and 'Year' in som_df.columns:
        som_df['Game_Age_Years'] = date.today().year - som_df['Year']
        st.write(f"'Game_Age_Years' column added based on the 'Year' column.")

    # Select categorical columns and apply one-hot encoding
    categorical_cols = ['Platform', 'Genre', 'Publisher']
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(som_df[categorical_cols]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

    # Combine the encoded columns with the original DataFrame
    som_df = pd.concat([som_df.drop(columns=categorical_cols), encoded_df], axis=1)

    # Define numerical columns
    numerical_cols = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Game_Age_Years']
    
    # Check if the necessary numerical columns exist in the DataFrame
    missing_columns = [col for col in numerical_cols if col not in som_df.columns]
    if missing_columns:
        st.error(f"The following columns are missing from the uploaded data: {', '.join(missing_columns)}")
        return  # Stop execution if required columns are missing
    
    # Scale numerical columns
    scaler = MinMaxScaler()
    som_df[numerical_cols] = scaler.fit_transform(som_df[numerical_cols])

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=12)
    pca_data = pca.fit_transform(som_df)

    # Initialize and train the SOM
    som = MiniSom(x=som_size, y=som_size, input_len=pca_data.shape[1], sigma=1.0, learning_rate=learning_rate)
    som.train_batch(pca_data, num_iterations)

    # Retrieve SOM clusters for each data point
    cluster_labels = [som.winner(x) for x in pca_data]
    cluster_labels_df = pd.DataFrame(cluster_labels, columns=['Cluster_X', 'Cluster_Y'])

    # Combine PCA data with cluster labels
    data_with_clusters = pd.concat([pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])]), cluster_labels_df], axis=1)
    data_with_clusters['Cluster'] = cluster_labels_df['Cluster_X'] * som_size + cluster_labels_df['Cluster_Y']

    # Plot PCA scatter with SOM clusters
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=data_with_clusters, x='PC1', y='PC2', hue='Cluster_X', palette='tab10', marker='o')
    plt.title('PCA Scatter Plot with SOM Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster_X', loc='best')
    st.pyplot(plt)

    # Plot U-Matrix
    weights = som.get_weights()
    u_matrix = np.zeros((som_size, som_size))
    for i in range(som_size):
        for j in range(som_size):
            if i < som_size - 1:
                u_matrix[i, j] += np.linalg.norm(weights[i, j] - weights[i + 1, j])
            if j < som_size - 1:
                u_matrix[i, j] += np.linalg.norm(weights[i, j] - weights[i, j + 1])
            u_matrix[i, j] /= 2
    plt.figure(figsize=(5, 5))
    plt.imshow(u_matrix, cmap='bone')
    plt.colorbar()
    plt.title('U-Matrix')
    st.pyplot(plt)

    # Plot Winner Map
    winner_map = np.zeros((som_size, som_size))
    for i, x in enumerate(pca_data):
        w = som.winner(x)
        winner_map[w[0], w[1]] += 1
    winner_map = winner_map / np.max(winner_map)
    plt.figure(figsize=(5, 5))
    plt.pcolor(winner_map, cmap='tab20', edgecolors='k')
    plt.colorbar()
    plt.title('Winner Map')
    st.pyplot(plt)

    # Silhouette Score calculation
    X = data_with_clusters[[f'PC{i+1}' for i in range(pca_data.shape[1])]].values
    labels = data_with_clusters['Cluster'].values
    silhouette_avg = silhouette_score(X, labels, metric='euclidean')
    st.write(f"Silhouette Score: {silhouette_avg}")

    # Silhouette Score histogram
    silhouette_vals = silhouette_samples(X, labels, metric='euclidean')
    plt.figure(figsize=(5, 5))
    plt.hist(silhouette_vals, bins=30, edgecolor='k', alpha=0.7)
    plt.title('Histogram of Silhouette Scores')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Streamlit app layout
st.title('Self-Organizing Maps (SOM) Clustering')

# File uploader for CSV data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Drop unnecessary columns
    som_df = df.drop(columns=['Name', 'Rank'], errors='ignore')

    # Display the first few rows of the DataFrame
    st.write("Here are the first few rows of your file:")
    st.write(som_df.head())

    # User inputs for SOM parameters
    som_size = st.slider('SOM Grid Size (NxN):', min_value=5, max_value=20, value=12)
    learning_rate = st.slider('Learning Rate:', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    num_iterations = st.slider('Number of Iterations:', min_value=1000, max_value=10000, value=5000, step=500)

    # Run SOM clustering and visualizations
    run_som(som_df, som_size, learning_rate, num_iterations)
else:
    st.write("Please upload a CSV file to proceed.")
