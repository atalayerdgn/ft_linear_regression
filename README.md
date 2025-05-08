# ft_linear_regression

A machine learning project that implements a simple linear regression model to predict car prices based on mileage.

## Overview

This project contains two main programs:
- **Training program**: Learns the relationship between car mileage and price from a dataset
- **Prediction program**: Predicts the price of a car based on its mileage using the trained model

## How It Works

### Linear Regression Model

The model predicts car prices using the equation:
```
estimatedPrice = theta0 + (theta1 * mileage)
```

Where:
- `theta0` and `theta1` are coefficients determined through training
- `mileage` is the input feature (kilometers driven)

### Training Algorithm

The training program uses gradient descent to minimize prediction error:
1. Initialize `theta0` and `theta1` to 0
2. For each iteration:
   - Calculate predicted prices using current coefficients
   - Update coefficients using the gradient descent formulas:
     ```
     tmpTheta0 = learningRate * (1/m) * sum(estimatedPrice[i] - price[i])
     tmpTheta1 = learningRate * (1/m) * sum((estimatedPrice[i] - price[i]) * mileage[i])
     ```
   - Apply updates to both coefficients simultaneously
   - Continue until convergence or maximum iterations reached

## Usage

### Training the Model
```
python train.py
```
This reads the dataset, trains the model, and saves the learned coefficients.

### Making Predictions
```
python predict.py
```
This loads the trained coefficients and prompts you to enter a mileage value to get a price estimate.

## Requirements

- Python 3.x
- NumPy (for numerical operations)
- Matplotlib (for visualization)

## Project Structure

- `train.py`: Training program implementation
- `predict.py`: Prediction program implementation
- `data.csv`: Dataset containing mileage and price information
- `model.json`: Stores trained model coefficients

## Limitations

- The model assumes a linear relationship between mileage and price
- Accuracy depends on dataset quality and size
- Simple model that doesn't account for other factors affecting car prices

## License

This project is part of the 42 School curriculum.
