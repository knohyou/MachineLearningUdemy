# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # upperbound not included. We want to keep in matrix form not a vector for indepednent variable??
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Polynomial Regression to the dataset
# Regressor would be the nonlinear regressor object. 

# Predict new result with Polynomial Regression
y2_pred = regressor.predict(6.5)


# Visualize the Polynomial Regression Results
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualize the Polynomial Regression Results for higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()