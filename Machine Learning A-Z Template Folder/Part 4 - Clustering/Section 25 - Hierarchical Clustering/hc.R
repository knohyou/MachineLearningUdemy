# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[,4:5]

# Using Dendogram to find optimal number of clusters
dendogram = hclust(dist(X, method = 'euclidean'),
                   method = 'ward.D')
plot(dendogram,
     main = paste('Dendogram'),
     xlab = 'Customers',
     ylab = 'Eucledian Distance')

# Fitting HC using the optimal cluster
hc = hclust(dist(X, method = 'euclidean'),
                   method = 'ward.D')
y_hc = cutree(hc, k = 5) # Dendogram hclust has the information of the hc

# Plotting the HC plot
clusplot(dataset, 
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels= 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Cluster Plot of Mall Customers'),
         xlab = 'Salary Income',
         ylab = 'Spending Score')