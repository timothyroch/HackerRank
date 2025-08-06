"""

Problem Summary (HackerRank):

This problem is based on real CBSE Class 12 exam data from 2013. Each student took 5 subjects,
including Mathematics. For each student, we are given their grades (1 to 8) in 4 subjects 
(English + 3 electives), and we must predict the missing Mathematics grade.

The grade distribution is such that 1 is the best and 8 is the lowest among passing students.
The goal is to predict the grade in Mathematics as accurately as possible, with predictions 
considered correct if they are within ±1 of the true value.

Approach I used:

1. Read Training Data: Load "training.json" containing student records with all 5 grades.
2. Group by Subject Combinations: Students with the same subject set (excluding Math) are grouped.
3. Feature Extraction:
   - Use basic grades from the 4 known subjects.
   - Add statistical features: total, average, max, min.
4. Train Linear Models:
   - For each subject combination, fit a linear regression model using normal equations.
   - Add regularization (Ridge regression) to ensure numerical stability.
   - If not enough data for a combination, fall back to using the average Mathematics grade.
5. Global Fallback Model:
   - A simple model trained on all students, using only 4 grades plus basic stats.
   - Used if no specific model exists for a student's subject combo.
6. Prediction Phase:
   - For each test case, predict the Mathematics grade using:
     a. The model for their subject combo, if it exists.
     b. Otherwise, the global model.
     c. Otherwise, the average of the student's 4 known grades.
   - Clamp predicted values to the valid range [1, 8].

My method uses a simple machine learning approach (linear regression) but adapts it
to account for the fact that different subject combinations may have different grading
patterns. It also handles missing or underrepresented combinations with reasonable fallbacks.

"""


import sys
import json
from collections import defaultdict

def get_subject_combination(record):
    """Identify which subject combination this record belongs to"""
    # Remove identifier fields (both SerialNumber and serial)
    subjects = set(record.keys()) - {'SerialNumber', 'serial', 'Mathematics'}
    return tuple(sorted(subjects))

def extract_features(record, subject_combination):
    """Extract features for a specific subject combination"""
    features = []
    subjects = sorted(subject_combination)
    
    # Basic grades
    grades = [record.get(subj, 4) for subj in subjects]  
    features.extend(grades)
    
    # Statistical features
    if len(grades) > 0:
        features.append(sum(grades))  
        features.append(sum(grades) / len(grades))  
        features.append(max(grades))  
        features.append(min(grades))  
    
    return features

# Simple linear regression with closed form solution
def train_linear_model(X, y):
    """Train linear regression using normal equations"""
    if len(X) == 0:
        return None
    
    # Add bias term
    n = len(X)
    m = len(X[0]) + 1
    
    # Create design matrix with bias
    A = []
    for i in range(n):
        row = [1.0] + X[i]
        A.append(row)
    
    # Compute A^T * A
    AtA = [[0.0] * m for _ in range(m)]
    for i in range(n):
        for j in range(m):
            for k in range(m):
                AtA[j][k] += A[i][j] * A[i][k]
    
    # Add regularization
    reg = 1e-6
    for i in range(m):
        AtA[i][i] += reg
    
    # Compute A^T * y
    Aty = [0.0] * m
    for i in range(n):
        for j in range(m):
            Aty[j] += A[i][j] * y[i]
    
    # Solve using Gaussian elimination
    try:
        weights = solve_system(AtA, Aty)
        return weights
    except:
        return None

def solve_system(A, b):
    """Solve linear system Ax = b using Gaussian elimination"""
    n = len(A)
    # Create augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        
        # Swap rows
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        # Eliminate
        for k in range(i + 1, n):
            if abs(aug[i][i]) > 1e-12:
                factor = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        if abs(aug[i][i]) > 1e-12:
            x[i] /= aug[i][i]
    
    return x

def predict(weights, features):
    """Make prediction using linear model"""
    if weights is None:
        return sum(features[:4]) / 4 if len(features) >= 4 else 4
    
    x = [1.0] + features  # Add bias
    return sum(w * f for w, f in zip(weights, x))

# Load and process training data
try:
    training_data = []
    with open("training.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if 'Mathematics' in record:
                        training_data.append(record)
                except:
                    continue
except:
    training_data = []

# Group by subject combination and train models
models = {}
combination_stats = defaultdict(list)

for record in training_data:
    combo = get_subject_combination(record)
    combination_stats[combo].append(record['Mathematics'])

for combo, math_grades in combination_stats.items():
    if len(math_grades) < 3:  # Need minimum samples
        continue
    
    # Get training data for this combination
    X_train = []
    y_train = []
    
    for record in training_data:
        if get_subject_combination(record) == combo:
            features = extract_features(record, combo)
            X_train.append(features)
            y_train.append(record['Mathematics'])
    
    if len(X_train) >= 3:
        weights = train_linear_model(X_train, y_train)
        if weights is not None:
            models[combo] = weights
        else:
            # Fallback to average
            models[combo] = sum(y_train) / len(y_train)

# Global fallback model
global_features = []
global_targets = []

for record in training_data:
    subjects = sorted(set(record.keys()) - {'SerialNumber', 'serial', 'Mathematics'})
    grades = [record.get(subj, 4) for subj in subjects]
    
    # Create standardized feature vector
    while len(grades) < 4:
        grades.append(4)
    grades = grades[:4]
    
    features = grades + [sum(grades), sum(grades)/len(grades)]
    global_features.append(features)
    global_targets.append(record['Mathematics'])

global_model = train_linear_model(global_features, global_targets) if global_features else None

# Process test cases
N = int(sys.stdin.readline())
for _ in range(N):
    try:
        line = sys.stdin.readline().strip()
        if not line:
            print(4)
            continue
            
        record = json.loads(line)
        combo = get_subject_combination(record)
        
        # Try combination-specific model
        if combo in models:
            if isinstance(models[combo], (int, float)):
                # Simple average fallback
                pred = models[combo]
            else:
                # Linear model
                features = extract_features(record, combo)
                pred = predict(models[combo], features)
        elif global_model:
            # Use global model
            subjects = sorted(set(record.keys()) - {'SerialNumber', 'serial'})
            grades = [record.get(subj, 4) for subj in subjects]
            while len(grades) < 4:
                grades.append(4)
            grades = grades[:4]
            features = grades + [sum(grades), sum(grades)/len(grades)]
            pred = predict(global_model, features)
        else:
            # Ultimate fallback: average of input grades
            grades = [v for k, v in record.items() if k not in ['SerialNumber', 'serial']]
            pred = sum(grades) / len(grades) if grades else 4
        
        # Clamp to valid range
        grade = max(1, min(8, round(pred)))
        print(grade)
        
    except Exception as e:
        # If anything goes wrong, output middle grade
        print(4)
