# Polynomial Linear Regression

# In the future, SVR, decision tree, random forest non-linear regressor


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # upperbound not included. We want to keep in matrix form not a vector for indepednent variable??
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
""" from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) """

# With a small dataset not split into training and test set. 
# Most accurate prediction need all dataset 

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X, y)
lin_reg = LinearRegression()
lin_reg.fit(X,y)



# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
# Transform the X into matrix of different orders of independent variables
# Automatically have the column of 1's 
# Need to incorporate this regressor into a linear model?? Create new linear regression object
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# Fit the linear regression to X poly??

# Visualize the Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualize the Polynomial Regression Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
# Increase resolution in the model 
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict new result with Linear Regression model
y1_pred = lin_reg.predict(6.5)

# Predict new result with Polynomial Regression
y2_pred = lin_reg_2.predict(poly_reg.fit_transform(6.5))



'''y_pred = regressor.predict(4.5)
plt.plot(X,y,'b-')
plt.plot(X, regressor.fit(X, y))
plt.plot(4.5, y_pred)
plt.show()'''