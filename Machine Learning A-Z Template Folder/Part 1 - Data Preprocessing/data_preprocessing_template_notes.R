# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age,FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary,FUN = function(x) mean(x,na.rm = TRUE)),
                        dataset$Salary)

# Encoding categorical data (More simple than Python) Factor Function
# Transform Cateogircal data into a column of factors. No need
# to create dummy variables

dataset$Country = factor(dataset$Country, 
                         levels = c('France','Spain','Germany'), # c is a vector. Create c vectory with France , S, G as elements
                         labels = c(1, 2, 3))
                         
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))
                         
# split the dataset into training and test set
# Instal library in R
#install.packages('bitops')
#install.packages('caTools')
library(caTools) # Incorporate the library in your code
set.seed(123) # Incoporate random
split = sample.split(dataset$Purchased, SplitRatio = 8/10) # Create the method to split the dataset
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
training_set[,2:3] = scale(training_set[,2:3]) # Problem Countires and Purchased are factors. Not numeric
test_set[,2:3] = scale(test_set[,2:3])

