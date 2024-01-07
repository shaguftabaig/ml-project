# Breast Cancer Classification
### Overview

This Python script is designed to solve the breast cancer classification problem using a dataset with features related to breast cancer diagnostics. The script uses Logistic Regression, a machine learning algorithm, to classify instances as either malignant (cancerous) or benign (non-cancerous).

### Dataset

The dataset used is based on the Breast Cancer Wisconsin (Diagnostic) Data Set. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

### Features

The dataset includes several features such as radius mean, texture mean, perimeter mean, area mean, smoothness mean, etc.
The target variable is 'diagnosis', where 'M' represents malignant and 'B' represents benign.

### Script Workflow
#### Data Preprocessing:

- Loading the dataset.
- Dropping unnecessary columns.
- Handling missing values.
- Encoding the target variable (diagnosis).
- Data Normalization and Splitting:
   - Normalizing the feature values.
   - Splitting the dataset into training (80%) and testing (20%) sets.
- Model Training and Evaluation:
   - Training a Logistic Regression model on the training data.
   - Evaluating the model on the test data using metrics like accuracy, precision, recall, F1-score, and a confusion matrix.
- Hyperparameter Tuning and Cross-Validation: 
   - Perform hyperparameter tuning using GridSearchCV to find the best parameters for the Logistic Regression model.
   - Employ cross-validation to assess the model's performance and ensure its stability and generalizability.
  
### Requirements
- Python 3.x
- Pandas
- Scikit-Learn
- Usage
- To run the script, ensure you have Python - installed along with the necessary libraries. Place the dataset (data.csv) in the same directory as the script and execute the script.

### Output
The script will output:

- The first few rows of the processed dataset.
- The shapes of the training and testing data.
- The accuracy of the model.
- A detailed classification report.
- A confusion matrix of the predictions.
- The best hyperparameters from the tuning process.
- The cross-validation accuracy scores.

### Author
Shagufta Afreen Baig