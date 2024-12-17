#!/usr/bin/env python3
"""
A simple example of a machine learning workflow using Python and scikit-learn.

This script:
- Loads the iris dataset
- Splits into train/test sets
- Trains a logistic regression model
- Evaluates and prints the model accuracy
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print("Test Accuracy:", accuracy)

    # Optional: Print some sample predictions for inspection
    print("Sample predictions:", y_pred[:5])
    print("Actual labels:    ", y_test[:5])


if __name__ == "__main__":
    main()
