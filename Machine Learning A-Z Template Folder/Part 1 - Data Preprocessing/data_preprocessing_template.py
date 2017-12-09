# Data Processing

# Import the library
import numpy as np #Statistical 
import matplotlib.pyplot as plt # Useful tool to plot 
import pandas as pd # Import and manage dataset

# Import Dataset
dataset = pd.read_csv('Data.csv')

#Create our independent and dependent variable
X = dataset.iloc[:,:-1].values 
Y = dataset.iloc[:,3].values

# Split dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Feature Scaling. Many machine learning algorithm based on ucledian distance
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """