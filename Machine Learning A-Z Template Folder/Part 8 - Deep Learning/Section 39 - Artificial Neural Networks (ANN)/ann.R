# Artificial Neural Network


# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[,4:14]

# Encoding the data categorical data
# Need to set as factors and numeric 
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# split the dataset into training and test set
#install.packages('bitops')
#install.packages('caTools')
library(caTools) 
set.seed(123) 
split = sample.split(dataset$Exited, SplitRatio = 8/10) # Create the method to split the dataset
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
training_set[,-11] = scale(training_set[,-11]) 
test_set[,-11] = scale(test_set[,-11])

# Fitting ANN to the Training Set
# h20 package open source. Connect to a h2o instance Provide options for model.
# Allows parameter tuning 
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1) # use all cores  Connect to h2o instance
classifier = h2o.deeplearning(y = 'Exited', 
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6,6), # 2 hidden layers and 6 nodes in each Number of hidden layer, 2nd parameter is number of nodes
                              epochs = 100,   # epochs number of dataset iterated
                              train_samples_per_iteration = -2) # batch size. parameter tuned already



# Predicting the Test set Results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
#y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred = prob_pred > 0.5
y_pred = as.vector(y_pred)


# Making the Confusion matrix
cm = table(test_set[,11], y_pred)

h2o.shutdown()
