# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Find optimal number of clusters using dendogram
import scipy.cluster.hierarchy as sch # Shortcut sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distances')
plt.show()
# Ward method minimie the variance within cluster

# Fit Hiearchical Clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,
                             affinity = 'euclidean',
                             linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualize the hierarchical Clustering
plt.scatter(X[y_hc == 0,0], X[y_hc == 0, 1], c = 'red', s = 100, label = 'Cluster1')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1, 1], c = 'blue', s = 100, label = 'Cluster2')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2, 1], c = 'green', s = 100, label = 'Cluster3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3, 1], c = 'cyan', s = 100, label = 'Cluster4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4, 1], c = 'magenta', s = 100, label = 'Cluster5')
plt.title('HC Clustering of Mall Clients')
plt.xlabel('Annual Salary')
plt.ylabel('Spending Score')
plt.legend()
plt.show()