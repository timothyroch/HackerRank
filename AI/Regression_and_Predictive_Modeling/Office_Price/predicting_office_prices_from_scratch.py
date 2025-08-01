"""

Polynomial Regression from Scratch — No Libraries

This implementation solves a prediction problem where the target variable
(price per square foot) is assumed to follow a multivariate polynomial
relationship with the input features (normalized between 0 and 1).

Problem:
Charlie wants to buy office space and has conducted a survey of the area. He has quantified and normalized various office space features,
and mapped them to values between 0 and 1. Each row in Charlie's data table contains these feature values followed by the price per 
square foot. However, some prices are missing. Charlie needs your help to predict the missing prices. The prices per square foot are 
approximately a polynomial function of the features, and you are tasked with using this relationship to predict the missing prices for
some offices. The prices per square foot, are approximately a polynomial function of the features in the observation table. This polynomial
always has an order less than 4.

Approach:
- Polynomial feature expansion up to degree 3
- Closed-form linear regression using normal equations
- Manual implementation of matrix operations (transpose, multiplication, inversion)
- Ridge regularization to ensure numerical stability

This solution does not use any external libraries (e.g. NumPy, sklearn).
All computations are implemented from scratch.

This code was originally designed to solve a HackerRank machine learning challenge,
but can be generalized for any small-scale polynomial regression task.


Sample Input

2 100
0.44 0.68 511.14
0.99 0.23 717.1
0.84 0.29 607.91
0.28 0.45 270.4
...
0.07 0.83 289.88
0.66 0.8 830.85
0.73 0.92 1038.09
0.77 0.36 593.86
4
0.05 0.54
0.91 0.91
0.31 0.76
0.51 0.31

"""

X_train = []
Y_train = []
X_test = []

def poly_features(f):
    features = [1.0]
    n = len(f)
    for i in range(n):
        features.append(f[i])
    for i in range(n):
        for j in range(i, n):
            features.append(f[i] * f[j])
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                features.append(f[i] * f[j] * f[k])
    return features

F, N = map(int, input().split())
for _ in range(N):
    raw = input().strip()
    if raw:
        *features, target = map(float, raw.split())
        X_train.append(poly_features(features))
        Y_train.append([target])

T = int(input())
for _ in range(T):
    features = list(map(float, input().split()))
    X_test.append(poly_features(features))

# Matrix operations

def transpose(matrix):
    """Transpose a matrix."""
    return [list(row) for row in zip(*matrix)]

def matmul(A, B):
    """Multiply two matrices A and B."""
    result = []
    for row in A:
        new_row = []
        for col in zip(*B):
            new_row.append(sum(a*b for a, b in zip(row, col)))
        result.append(new_row)
    return result

def inverse(matrix):
    """Compute the inverse of a square matrix using Gaussian elimination."""
    n = len(matrix)
    A = [row[:] for row in matrix]
    I = [[float(i == j) for i in range(n)] for j in range(n)]
    for i in range(n):
        factor = A[i][i]
        if factor == 0:
            raise  ValueError("Matrix is singular (non-invertible)")
        for j in range(n):
            A[i][j] /= factor
            I[i][j] /= factor
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(n):
                    A[k][j] -= factor * A[i][j]
                    I[k][j] -= factor * I[i][j]
    return I

def train_linear_regression(x, y, lambda_=1e-8):
    X_T = transpose(x)
    XTX = matmul(X_T, x)
    for i in range(len(XTX)):
        XTX[i][i] += lambda_
    XTX_inv = inverse(XTX)
    return matmul(XTX_inv, matmul(X_T, y))

def predict(x, beta):
    return matmul(x, beta)

# Train the model
beta = train_linear_regression(X_train, Y_train)
predictions = predict(X_test, beta)

# Output
for res in predictions:
    print(f"{res[0]:.2f}")
