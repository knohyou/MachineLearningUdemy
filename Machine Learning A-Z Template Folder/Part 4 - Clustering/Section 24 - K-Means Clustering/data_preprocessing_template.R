# KMeans Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[,4:5]

# Using Elbow method to obtain optimal 
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(X,i)$withinss)
# Kmeans retreans an object class kmeans
# withinss = Vector of within-cluster sum of squares, one component per cluster.
plot(1:10, wcss, 
     type = 'b', 
     main = paste('Cluster of Clients'), 
     xlab = 'Number of Clusters', 
     ylab = 'WCSS')

# Apply K Means to the dataset
set.seed(29)
kmeans = kmeans(X, centers = 5, iter.max = 300, nstart = 10)

# Visualize the Clusters
library(cluster)
clusplot(X, 
         kmeans$cluster,
         line = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of Clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')
