import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Dataset provided
data = {
    'X': [1.613, 0.280, 0.253, 0.940, 1.486, 1.196, 1.166, 1.540, 0.359],
    'Y': [1.526, 1.706, 1.140, 1.566, 0.559, 1.086, 1.106, 0.429, 1.799],
    'Target': [0, 1, 1, 0, 1, 0, 0, 1, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Prepare X (features) and y (target)
X = df[['X', 'Y']].values
y = df['Target'].values

# KMeans clustering with 3 centroids
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# New point to classify (X=0.903, Y=0.601)
new_point = np.array([[0.903, 0.601]])

# Find the closest centroid
closest_centroid_idx = kmeans.predict(new_point)
centroids = kmeans.cluster_centers_

# Calculate the Euclidean distance from the new point to each centroid
distances = np.linalg.norm(centroids - new_point, axis=1)

# Find the classification of the closest centroid
closest_centroid_class = pairwise_distances_argmin_min(new_point, centroids)[0]
predicted_class = y[closest_centroid_class]

# Output the results
print(f"Centroids: {centroids}")
print(f"Predicted classification for point (X=0.903, Y=0.601): {predicted_class}")
