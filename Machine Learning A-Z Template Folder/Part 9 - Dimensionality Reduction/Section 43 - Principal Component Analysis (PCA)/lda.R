# lda Linear Discriminant Analysis

# Importing the dataset
dataset = read.csv('Wine.csv')


# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[, -14] = scale(training_set[, -14])
test_set[, -14] = scale(test_set[, -14])

# Apply LDA
#install.packages('MASS')
library(MASS)
lda = lda(formula = Customer_Segment ~., 
          data = training_set)
# Number of linear discrimant k-1. 3 classes have 2 linear discriminant 
training_set = as.data.frame(predict(lda, training_set)) # Need to make sure we have a dataframe for the training set
training_set = training_set[c(5,6,1)]
test_set = as.data.frame(predict(lda, test_set))
test_set = test_set[c(5,6,1)]


#install.packages('e1071')
library(e1071)

# Fitting the Regression Model to the dataset
# Create your regressor here
classifier = svm(formula = class ~., 
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')
summary(classifier)

# Predicting a new result
y_pred = predict(classifier, type = 'response', test_set[-3]) 

#We are using the test set observation 

# Making the Confusion Matrix
cm = table(test_set[,3], y_pred)

# Visualising the Training Set Model results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('x.LD1','x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[,-3], 
     main = 'Logistic Regression (Training Set)',
     xlab = 'LD1',
     ylab =  'LD2',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[,3] == 2, 'blue3', ifelse(set[,3] == 1, 'green4', 'red3')))

# Visualising the Test Set Model results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
X2 = seq(min(set[,2])-1, max(set[,2])+1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('x.LD1','x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[,-3], 
     main = 'Logistic Regression (Training Set)',
     xlab = 'LD1',
     ylab =  'LD2',
     xlim = range(X1),
     ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[,3] == 2, 'blue3', ifelse(set[,3] == 1, 'green4', 'red3')))
