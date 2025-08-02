"""
Polynomial Regression with Libraries (scikit-learn, numpy)

This implementation solves the same problem as the manual version,
but leverages established Python libraries for numerical computing
and machine learning.

Problem:
Charlie wants to buy office space and has collected normalized feature
data for several office locations, along with the corresponding prices
per square foot. Some prices are missing. The objective is to predict
these missing prices using a polynomial regression model.

Approach:
- Uses scikit-learn's `PolynomialFeatures` to generate polynomial terms up to degree 3
- Trains a Ridge Regression model (L2-regularized linear regression) to reduce overfitting
- Performs predictions on new office data based on the learned model

"""

import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Read and store training data
X_train = []
Y_train = []

F, N = map(int, input().split()) # F: number of features, N: number of training examples
for _ in range(N):
    values = list(map(float, input().split()))
    X_train.append(values[:-1])
    Y_train.append(values[-1])
    
T = int(input())
X_test = [list(map(float, input().split())) for _ in range(T)]

# Convert lists to numpy arrays for use with scikit-learn
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

# Generate polynomial features up to degree 3 (with bias term)
poly = PolynomialFeatures(degree=3, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Ridge Regression model with small regularization to ensure stability
model = Ridge(alpha=1e-8, fit_intercept=False) 
model.fit(X_train_poly, Y_train)

# Predict on test data and print results
y_pred = model.predict(X_test_poly)
for y in y_pred:
    print(f"{y:.2f}")
