# Logistic Regression Implementation

This repository contains a Python implementation of logistic regression from scratch. Logistic regression is a fundamental machine learning algorithm used for binary classification tasks.

## Dependencies
- numpy
- pandas
- matplotlib
- tqdm
- scikit-learn

You can install the dependencies using pip:

```bash
pip install numpy pandas matplotlib tqdm scikit-learn
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your_username/your_repo.git
Navigate to the repository:
bash
Copy code
cd your_repo
Run the logistic_regression.py script:
bash
Copy code
python logistic_regression.py
Code Overview
The LogisticRegression class is implemented in logistic_regression.py. Below is a brief overview of its functionalities:

__init__: Initializes the logistic regression model with hyperparameters such as learning rate, tolerance, regularization, lambda parameter, and maximum iterations.
datasetReader: Loads the Breast Cancer dataset and splits it into training and testing sets.
normalize_data: Normalizes the data by subtracting the mean and dividing by the standard deviation.
addX0: Adds a column of ones to the feature matrix for the bias term.
sigmoid: Computes the sigmoid function.
decision_boundary: Plots the decision boundary for two-dimensional datasets.
costFunction: Computes the logistic regression cost function.
gradient: Computes the gradient of the cost function.
gradientDescent: Performs gradient descent optimization to minimize the cost function.
plotCost: Plots the cost function over iterations.
predict: Predicts the class labels for input data.
evaluate: Evaluates the model's performance using accuracy, precision, and recall metrics.
fit: Fits the logistic regression model to the training data, evaluates its performance on the test data, and plots the decision boundary.
Example
python
Copy code
from logistic_regression import LogisticRegression

# Initialize the model
model = LogisticRegression(learningRate=0.1, tolerance=1e-5, regularization=True, lambda_param=0.01)

# Fit the model
model.fit()
