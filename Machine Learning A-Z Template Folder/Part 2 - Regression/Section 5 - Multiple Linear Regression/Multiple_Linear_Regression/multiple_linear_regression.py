# Data Processing

# Import the library
import numpy as np #Statistical 
import matplotlib.pyplot as plt # Useful tool to plot 
import pandas as pd # Import and manage dataset

# Import Dataset
dataset = pd.read_csv('50_Startups.csv')

#Create our independent and dependent variable
X = dataset.iloc[:,:-1].values 
Y = dataset.iloc[:,4].values

#Encode categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy variable trap
X = X[:, 1:] # To remove any redundant dependency 


# Split dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Feature Scaling. Many machine learning algorithm based on ucledian distance
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """

# Fit Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Plotting the training set
plt.plot(list(range(10)),Y_test, 'bo', list(range(10)),y_pred, 'ro')
plt.xlabel('Indices')
plt.ylabel('Profit')
plt.show()

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm # Statsmodel doesn't take into account the constant
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) # Need to include a constant b0 = 1 column into matrix of features
# astype(int) prevent error so datatype is the same 
# Need to add the column at the beginning
X_opt = X[:,[0, 1, 2, 3, 4, 5]] # Final matrix only independent variables that are high impact

# Create new regressor of statsmodel New object
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
# Remove predictor x2 Remove column with highest p value

# Next iteration
X_opt = X[:,[0, 1, 3, 4, 5]] # Final matrix only independent variables that are high impact
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0, 3, 4, 5]] # Final matrix only independent variables that are high impact
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0, 3, 5]] # Final matrix only independent variables that are high impact
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0, 3]] # Final matrix only independent variables that are high impact
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()











