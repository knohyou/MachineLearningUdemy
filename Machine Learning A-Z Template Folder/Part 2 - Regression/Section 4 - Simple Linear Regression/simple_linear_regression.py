# Simple Linear Regression

# Import the library
import numpy as np #Statistical 
import matplotlib.pyplot as plt # Useful tool to plot 
import pandas as pd # Import and manage dataset

# Import Dataset
dataset = pd.read_csv('Salary_Data.csv')

#Create our independent and dependent variable
X = dataset.iloc[:,:-1].values 
Y = dataset.iloc[:,1].values

# Split dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

# Feature Scaling. Many machine learning algorithm based on ucledian distance
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # To call the class need to call
regressor.fit(X_train, Y_train)

# Predicting the Test Set results
Y_pred = regressor.predict(X_test)

#Visualizing the Training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), 'blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), 'blue')
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()