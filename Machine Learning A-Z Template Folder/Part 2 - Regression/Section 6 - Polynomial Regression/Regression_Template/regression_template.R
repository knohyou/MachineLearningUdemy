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


# Fitting to Polynomial
#Create Regression model

# predit Polynomial Regression
y_pred_poly = predict(regressor, data.frame(Level = 6.5))

# Plot Polynomial
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y=dataset$Salary), colour = 'red') + 
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Regression model)') +
  xlab('Levels') + 
  ylab('Salary')

# Visualize with More resolution
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y=dataset$Salary), colour = 'red') + 
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), colour = 'blue') +
  ggtitle('Truth or Bluff (Regression model)') +
  xlab('Levels') + 
  ylab('Salary')


