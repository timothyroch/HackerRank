"""

Charlie is in the market for a house and has gathered data on several properties, including a set of
feature values (measured on a consistent scale) for each house. For some of these houses, he also
knows the price per square foot, but others are missing this price data. Each entry includes a fixed
number of features followed by the price, resulting in a dataset where each row has one more
column than the number of features. Given that the price per square foot is approximately a linear
function of these features, my task is to use the available data to predict the missing prices using a
regression-based approach.

Sample Input:
--------------
2 7
0.18 0.89 109.85
1.0  0.26 155.72
0.92 0.11 137.66
0.07 0.37 76.17
0.85 0.16 139.75
0.99 0.41 162.6
0.87 0.47 151.77
4
0.49 0.18
0.57 0.80
0.56 0.64
0.76 0.18

Explanation:
- F = 2 (features)
- N = 7 (training examples)
- Each of the next 7 lines: 2 feature values + 1 price
- T = 4 (test cases)
- Each of the next 4 lines: 2 feature values (predict the price)

Expected Output:
--------------
105.22
142.68
132.94
129.71

To solve this problem, I used multiple linear regression based on the normal equation. First, I
structured the training data into a matrix X and a vector y, where each row in X contained the
house's features plus a bias term (1.0), and y contained the corresponding prices. I computed the
regression coefficients using the formula: beta = inverse(transpose(X) * X) * transpose(X) * y.
This gives the set of weights that minimizes the squared error between predicted and actual prices.
Then, for each test house, I constructed a row with its features and a bias term, and calculated the
predicted price using: prediction = test_row * beta.


"""

# Read input data
X_train = []
y_train = []
X_test = []

F, N = map(int, input().split())
for _ in range(N):
  *features, target = map(float, input().split())
  X_train.append(features + [1.0])
  y_train.append([target])  
        
T = int(input())
for _ in range(T):
  features = list(map(float, input().split()))
  X_test.append(features + [1.0])  


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
            new_row.append(sum(a * b for a, b in zip(row, col)))
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
            raise ValueError("Matrix is singular (non-invertible)")
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

# Linear regression via normal equation
def train_linear_regression(X, y):
    X_T = transpose(X)
    XTX_inv = inverse(matmul(X_T, X))
    return matmul(XTX_inv, matmul(X_T, y))

def predict(X, beta):
    return matmul(X, beta)

# Train the model
beta = train_linear_regression(X_train, y_train)

# Predict prices
predictions = predict(X_test, beta)

# Output
for p in predictions:
    print(f"{p[0]:.2f}")
