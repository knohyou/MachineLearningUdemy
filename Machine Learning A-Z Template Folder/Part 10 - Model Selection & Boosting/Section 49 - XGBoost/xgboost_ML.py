# XGBoost

# Install xgboost following the instructions on this link

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

# With xgboost, the code already performs the feature selection
from xgboost import XGBClassifier
classifier = XGBClassifier()
# n_estimators = 100 number of trees
classifier.fit(X_train, y_train)

# Predicint Test set results
y_pred = classifier.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Apply k-fold cross validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10) # 10 fold cross validation
# if you have lot of dataset, n_jobs = -1 to use all CPU's 
accuracies.mean()
accuracies.std()

