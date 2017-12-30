# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Fast for numerical computation
# Run on GPU and CPU. More powerful more floating points. Use with numpy. Run parallel 
# Simple neural network can just use CPU

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
# Run both GPU and CPU
 
# Both Theano and tensorflow used for research purposes

# Installing Keras
# pip install --upgrade keras
# Keras a library based on Theano and Tensorflow
# Efficiently create ANN


# Part1 - Data Preprocessing
# Import the library
import numpy as np #Statistical 
import matplotlib.pyplot as plt # Useful tool to plot 
import pandas as pd # Import and manage dataset

# Import Dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Create our independent and dependent variable
X = dataset.iloc[:,3:13].values 
y = dataset.iloc[:,13].values

# Encoding categorical independent data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Encode the independent data to each column
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] # To remove a dummy variable 


# Split dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Feature Scaling. Many machine learning algorithm based on ucledian distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the training set
# Create classifier here
import keras
# Sequential Module
from keras.models import Sequential
# Dense Module
from keras.layers import Dense
# Dense function step1 selects random initial points close to 0 
# Each feature in one input node
# 11 features in one input node

# Choose rectifier function for hidden layer
# Choose signmoid activtation function for output layer


# Initiale ANN by defining as sequence of layers
classifier = Sequential()

# Adding first hidden layer
classifier.add(Dense(units = 6, use_bias=True, kernel_initializer ='glorot_uniform',  bias_initializer='zeros', activation = 'relu', input_dim =11))
# How many nodes to choose in hiddne layer?
# Take average of input and output node 
# (11+1)/2
#classifier.add(Dense(units = 6, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu' , input_dim =11) )
# units is the number of output 
# Input layer is 11
# relu - rectifier function

# Add second hidden layer
classifier.add(Dense(units = 6, use_bias=True, kernel_initializer ='glorot_uniform',  bias_initializer='zeros', activation = 'relu'))

# Final Output layer
classifier.add(Dense(units = 1, use_bias=True, kernel_initializer ='glorot_uniform',  bias_initializer='zeros', activation = 'sigmoid'))
# Want a probability activitation function to sigmoid function
# if you have 3 dependent categories, change units = 3 and activation = 'softmax???'

# Compile Neural network  ANN
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
# stochastic gradient efficient one - adam algorithm
# logarithmic loss function more than 3 categorical  categorical_entropy 

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Part 3 - Making predictions and evaluate the model

# Predicint Test set results
y_pred_original = classifier.predict(X_test)

y_pred = (y_pred_original > 0.5)
# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
