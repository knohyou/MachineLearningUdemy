# Polynomial Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
  
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
'''library(caTools)
set.seed(123)
split = sample.split(dataset$DependentVariable, SplitRatio = 1)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)'''


# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting to Linear Regression 
#regressor = lm(forumula = Salary ~ ., 
#               data = dataset)
regressor = lm(formula = Salary ~ Level, 
              data = dataset)
summary(regressor)

# Fitting to Polynomial
dataset$Level2 = dataset$Level^2
dataset$Level3= dataset$Level^3 
dataset$Level4= dataset$Level^4 
regressor_poly = lm(formula = Salary ~ .,
                    data = dataset)
#regressor_poly = lm(formula = Salary ~ poly(Level, degree = 4),
#                    data = dataset)
summary(regressor_poly)

# Plot the Linear Regression
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y=dataset$Salary), colour = 'red') + 
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regresssion)') +
  xlab('Levels') + 
  ylab('Salary')
  
# Plot Polynomial
ggplot() + 
  geom_point(aes(x = dataset$Level, y=dataset$Salary), colour = 'red') + 
  geom_line(aes(x = dataset$Level, y = predict(regressor_poly, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regresssion)') +
  xlab('Levels') + 
  ylab('Salary')

# predit Linear Regression
y_pred_lin = predict(regressor, data.frame(Level = 6.5))

# predit Polynomial Regression
y_pred_poly = predict(regressor_poly, data.frame(Level = 6.5, 
                                                 Level2 = 6.5^2, 
                                                 Level3 = 6.5^3, 
                                                 Level4 =6.5^4))
