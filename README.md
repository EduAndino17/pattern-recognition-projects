This program uses the K-means clustering algorithm to classify a set of 2D points into a specified number of clusters:

```
Copy code
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate some random 2D points
num_points = 1000
X = np.random.rand(num_points, 2)

# Set the number of clusters
num_clusters = 5

# Initialize the K-means model
kmeans = KMeans(n_clusters=num_clusters)

# Fit the model to the points
kmeans.fit(X)

# Predict the cluster labels for each point
labels = kmeans.predict(X)

# Plot the points with colors corresponding to their cluster label
colors = ["r", "g", "b", "y", "c"]
for i in range(num_points):
    plt.scatter(X[i, 0], X[i, 1], c=colors[labels[i]])
plt.show()

```
This code generates a set of random 2D points, initializes a K-means model with a specified number of clusters, fits the model to the points, and predicts the cluster labels for each point. It then plots the points with colors corresponding to their cluster labels. You can modify this code by changing the number of clusters or the way that the points are generated to suit your needs.
