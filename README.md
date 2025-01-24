# FT_LINEAR_REGRESSION

This project implements a basic linear regression model to estimate the price of a car based on its mileage.

# Components:

## Prediction Program:

Prompts the user for a car's mileage.

Calculates the estimated price using the following formula:

### estimatedPrice = theta0 + (theta1 * mileage)

theta0 and theta1 are the model's coefficients, initially set to 0.

## Training Program:


Reads a dataset containing car mileage and corresponding prices.

Performs linear regression to find the optimal values of theta0 and theta1 that minimize the prediction error.

Uses the following gradient descent update rules:

### tmpTheta0 = learningRate * (1 / m) * sum(estimatedPrice[i] - price[i])

### tmpTheta1 = learningRate * (1 / m) * sum((estimatedPrice[i] - price[i]) * mileage[i])

m is the number of data points in the dataset.

learningRate controls the speed of convergence (typically a small value).

Updates theta0 and theta1 simultaneously based on these calculations.

## Implementation Details:


The specific programming language and libraries used will depend on your preference. Popular choices include Python with libraries like NumPy, Scikit-learn, or TensorFlow.

The dataset should be formatted appropriately, with columns for mileage and price.

## Usage:


Train the model using the training program on your car mileage dataset.

Run the prediction program to estimate the price of a car by entering its mileage.

## Note:


The accuracy of the model depends on the quality and size of your dataset.

Simple linear regression assumes a linear relationship between mileage and price, which may not always hold true in real-world scenarios.

Further Considerations:


You can explore more advanced regression techniques like polynomial regression or decision trees for potentially better accuracy.

Implement error handling and data validation to ensure the robustness of your programs.
