# Principal-Component-Analysis-PCA-
Logistic Regression with PCA on Cancer Dataset
Overview

This project demonstrates the implementation of Principal Component Analysis (PCA) for dimensionality reduction and logistic regression for classification using the breast cancer dataset from sklearn.datasets. The objective is to identify essential features, reduce the dataset into two principal components, and perform logistic regression to classify malignant and benign tumors.
Project Files

    cancer_pca_logistic.py: Python script containing the code for PCA and logistic regression.
    README.md: This file provides instructions for understanding and running the project.
    requirements.txt: List of Python dependencies required to run the project.

Setup Instructions

Follow these steps to set up and run the project:
1. Prerequisites

Ensure you have Python 3.8+ installed on your system. You also need the following libraries:

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn

Install these libraries using the command:

pip install -r requirements.txt

2. Dataset

The project uses the breast cancer dataset from sklearn.datasets. This dataset is automatically loaded by the script using the load_breast_cancer() function, so no manual download is required.
Step-by-Step Instructions
1. Load the Dataset

The dataset is loaded using load_breast_cancer() from sklearn.datasets. It contains information about tumors (malignant or benign) with features describing characteristics of cell nuclei.

Key Steps:

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

2. Standardize the Data

Standardization is performed to ensure that all features have a mean of 0 and standard deviation of 1, which is essential for PCA.

Key Steps:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

3. Apply PCA

Reduce the dataset to 2 principal components to simplify the data while retaining maximum variance.

Key Steps:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

You can visualize the PCA-transformed data using a scatterplot:

import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Breast Cancer Data')
plt.colorbar(label='Tumor Type')
plt.show()

4. Train-Test Split

Split the PCA-transformed data into training and testing datasets.

Key Steps:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

5. Logistic Regression

Train a logistic regression model on the PCA-transformed training data and evaluate its performance on the test data.

Key Steps:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy}")
print("Classification Report:\n", report)

6. Results

The model achieved an accuracy of 99.12% on the test data. Below is the classification performance:

    Precision: 1.00 for malignant and 0.99 for benign.
    Recall: 0.98 for malignant and 1.00 for benign.
    F1-score: 0.99 for both classes.

7. Bonus (Optional): Visualization of PCA Components

Visualize how the PCA components separate malignant and benign tumors in a 2D space.

Key Steps:

import seaborn as sns
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Components with Tumor Classes')
plt.show()

How to Run the Code

    Clone the repository or unzip the project files.
    Navigate to the project directory.
    Run the Python script:

    python cancer_pca_logistic.py

    Review the outputs in the terminal and any visualizations displayed.

Requirements

The project requires the following Python libraries:

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn

Ensure these dependencies are installed by running:

pip install -r requirements.txt

Project Structure

/project_directory
|-- cancer_pca_logistic.py   # Main script
|-- requirements.txt         # Required dependencies
|-- README.md                # Instructions and details
