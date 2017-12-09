# Data Processing


# Import the library
import numpy as np #Statistical 
import matplotlib.pyplot as plt # Useful tool to plot 
import pandas as pd # Import and manage dataset

# Import Dataset
dataset = pd.read_csv('Data.csv')

#Create our independent variable
X = dataset.iloc[:,:-1].values # Take all columns except one

# Create Dependent Variable
Y = dataset.iloc[:,3].values

##########################
#Take care of missing data
from sklearn.preprocessing import Imputer 
#Contain 
# Imputer class

imputer = Imputer(missing_values="NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding Categorical Data (Country has 3 and Output has 2)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #Import LabelEncoder class from sklearn.preprocessing Python Module
labelencoder_X = LabelEncoder() #Define the labelencoder object
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #Apply fit_transform method to labelencoder object

###########################
# To prevent machine learning from thinking Germany is greater value than
# Spain, create dummy variables. Dummy encoding. 
#Prevent machine learning from introducing order to
# categorical variables
onehotencoder = OneHotEncoder(categorical_features = [0]) 
X = onehotencoder.fit_transform(X).toarray()

# For dependent variables, machine learning knows
# The categorical variables dont have order
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Split dataset into training and test set
# Cross validation library
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

########################
# Feature Scaling. Many machine learning algorithm based on ucledian distance
# Eucledian range is larger for salary and will dominate the algorithm
# Standardization vs Normalization Difference
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # no need to fit for test set
# Do you scale dummy variables?? Depends on if you want to keep 
# you interpretation. 
# Not need to do feature scaling for Y dependent variable because
# This is a classification problem with categorical 
# But when you do regression need to apply feature scaling
