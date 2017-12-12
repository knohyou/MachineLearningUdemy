# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Categorical Independent variable
# Encoding categorical data (More simple than Python) Factor Function
# Transform Cateogircal data into a column of factors. No need
# to create dummy variables


dataset$State = factor(dataset$State, 
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3),
                       ordered = FALSE)


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
# Profit is a linear combination of all the variables
regressor = lm(formula = Profit ~ .,
               data = training_set) # . indicates all independent variable
summary(regressor)
# The library knew it had to create a dummy variable 
# R removed the State1 dummy variable
# P value lower p value more statiscally differeent 
# if P value lower than 0.05 or 5% than variable is highly statistcally siginficant
# Only strong predictor is R&D spend on profit

#regressor = lm(formula = Profit ~ R.D.Spend,
#               data = training_set) # . indicates all independent variable
#summary(regressor)


# Predict the Test Set Result
Y_pred = predict(regressor, newdata = test_set)
# With which regressor to predict test set
# new data = new set of data to predict the profit

# Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
              data = dataset) # Just use complete dataset to see which variables is important
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = dataset) # Just use complete dataset to see which variables is important
summary(regressor)


regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
               data = dataset) # Just use complete dataset to see which variables is important
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend, 
               data = dataset) # Just use complete dataset to see which variables is important
summary(regressor)






