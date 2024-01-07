import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#---Data Preprocessing---
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Dropping unnecessary columns
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Checking for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values[missing_values > 0])

# Encoding the 'diagnosis' column: Malignant (M) as 1 and Benign (B) as 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B':0})

# Displaying the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(data.head())

#---Normalization and Data Splitting---
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Normalizing the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Displaying the shapes of the splits
print("\nShapes of the Training and Testing Sets:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#--Logical Regression---
# Initialise logical regression model
log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Logistic Regression Model Evaluation")
print(f"Accuracy: {accuracy:.2f}%")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", confusion_mat)

#range of hyperparameters for tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'solver': ['lbfgs', 'liblinear']
}

# GridSearchCV with cross-validation
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Training with the best parameters
log_reg = LogisticRegression(**best_params)
log_reg.fit(X_train, y_train)

# Cross-validation scores
cv_scores = cross_val_score(log_reg, X_normalized, y, cv=5) * 100
print("Cross-Validation Accuracy Scores:")
for i, score in enumerate(cv_scores):
    print(f" Fold {i + 1}: {score:.2f}%")
print(f"Mean CV Accuracy: {cv_scores.mean():.2f}%")