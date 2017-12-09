# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[,2:3]



# split the dataset into training and test set
#install.packages('bitops')
#install.packages('caTools')
library(caTools) 
set.seed(123) 
split = sample.split(dataset$Salary, SplitRatio = 2/3) # Create the method to split the dataset
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3]) 
# test_set[,2:3] = scale(test_set[,2:3])

# Fitting Simple Linear Regression to the Traiing Set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set) # salary is proporation al to Years of Experience
# summary(regressor)

# Predict the Test set results
Y_pred = predict(regressor, newdata = test_set)

# Visualizing the Trainig Set Results
#install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
              colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
              colour = 'blue') +
  ggtitle('Salary vs. Experience (Training Set)') + 
  xlab('Years of Experience') + 
  ylab('Salary')

            
# scatter plot of the training 

# Visualize the Test Set Results
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
           colour = 'blue') + 
  ggtitle('Salary vs. Experience (Testing Set)') + 
  xlab('Years of Experience') + 
  ylab('Salary')
  
