# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')
dataset = dataset[,2:3]


                         
# split the dataset into training and test set
#install.packages('bitops')
#install.packages('caTools')
library(caTools) 
set.seed(123) 
split = sample.split(dataset$Purchased, SplitRatio = 8/10) # Create the method to split the dataset
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3]) 
# test_set[,2:3] = scale(test_set[,2:3])

