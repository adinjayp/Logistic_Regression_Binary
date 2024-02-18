# Logistic Regression with Gradient Descent in Python

This repository contains an implementation of logistic regression with gradient descent in Python. The code includes data preprocessing, model training, prediction, and evaluation.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Classes](#classes)
  - [Examples](#examples)

## Installation
To use the code, you need to have Python and the following libraries installed:

- numpy
- pandas
- matplotlib
- tqdm
- sklearn

You can install them using pip:

```bash
pip install numpy pandas matplotlib tqdm scikit-learn
Usage
To use the logistic regression model, simply create an instance of the LogisticRegression class and call the fit() method.

python
Copy code
model = LogisticRegression(learningRate=0.1, tolerance=1e-5, regularization=True, lambda_param=0.01)
model.fit()
Classes
LogisticRegression: The LogisticRegression class implements the logistic regression model with gradient descent.

Parameters:

learningRate: float, optional (default=0.1) - The learning rate for gradient descent.
tolerance: float, optional (default=1e-5) - The tolerance for the convergence of gradient descent.
regularization: bool, optional (default=False) - Whether to use regularization or not.
lambda_param: float, optional (default=0.1) - The regularization parameter (Lambda).
maxIteration: int, optional (default=50000) - The maximum number of iterations for gradient descent.
Methods:

datasetReader(): returns the breast cancer dataset.
normalize_data(X): normalizes the input data.
normalize_train_test_data(X_train, X_test): normalizes the training and test data.
addX0(X): adds a column of ones to the input data.
sigmoid(z): calculates the sigmoid function.
decision_boundary(X, w): plots the decision boundary.
costFunction(X, y): calculates the cost function.
gradient(X, y): calculates the gradient.
gradientDescent(X, y): performs gradient descent.
plotCost(error_sequences): plots the cost function.
predict(X): predicts the classes.
evaluate(y, y_hat): evaluates the model performance.
fit(): trains the model.
Examples
Here are some examples of using the LogisticRegression class:

python
Copy code
# Example 1: Logistic Regression with Regularization
model = LogisticRegression(learningRate=0.1, tolerance=1e-5, regularization=True, lambda_param=0.01)
model.fit()

# Example 2: Logistic Regression without Regularization
model2 = LogisticRegression(learningRate=0.1, tolerance=1e-5, regularization=False, lambda_param=0.1)
model2.fit()
