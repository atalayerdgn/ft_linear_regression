# ft_linear_regression

A Python implementation of linear regression using both analytical (Normal Equations) and iterative (Gradient Descent) methods for predicting car prices based on mileage.

## Overview

This project implements linear regression to predict car prices based on their mileage (kilometers driven). It provides two different approaches:

1. **Normal Equations** - Analytical solution that directly computes optimal parameters
2. **Gradient Descent** - Iterative optimization method that gradually finds optimal parameters

## Features

- Data normalization to prevent numerical overflow
- Both analytical and iterative solutions
- Progress tracking during gradient descent
- Automatic parameter denormalization
- Model weights saving to file

## Requirements

- Python 3.x
- NumPy
- Pandas

Install dependencies:
```bash
pip install numpy pandas
```

## Usage

### Training the Model

Run the training script with your dataset:

```bash
python3 train.py data.csv
```

### Expected Output

```
Iteration 0: Cost = 0.500000
Iteration 100: Cost = 0.234567
Iteration 200: Cost = 0.123456
...
Iteration 900: Cost = 0.001234
Theta 0: 8484.761234, Theta 1: -0.021234
```

The trained parameters will be saved to `weights.txt`.

## File Structure

```
ft_linear_regression/
├── train.py       # Main training script
├── predict.py     # Prediction script (if available)
├── data.csv       # Training dataset
├── weights.txt    # Saved model parameters (generated)
└── README.md      # This file
```

## Dataset Format

The input CSV file should contain two columns:
- `km` - Mileage in kilometers
- `price` - Car price

Example:
```csv
km,price
240000,3650
139800,3800
150500,4400
...
```

## Algorithm Details

### Linear Regression Model

The model predicts price using the linear equation:
```
price = θ₀ + θ₁ × mileage
```

Where:
- `θ₀` (theta0) - Intercept parameter
- `θ₁` (theta1) - Slope parameter

### Normal Equations Method

Computes optimal parameters directly using:
```python
θ₁ = (n×Σ(xy) - Σ(x)×Σ(y)) / (n×Σ(x²) - (Σ(x))²)
θ₀ = (Σ(y) - θ₁×Σ(x)) / n
```

### Gradient Descent Method

Updates parameters iteratively using:
```python
tmp_θ₀ = learning_rate × (1/m) × Σ(errors)
tmp_θ₁ = learning_rate × (1/m) × Σ(errors × mileage)
```

Where:
- `learning_rate` - Step size for parameter updates (default: 0.01)
- `errors` - Difference between predicted and actual prices
- `m` - Number of training examples

## Data Normalization

To prevent numerical overflow with large mileage values, the algorithm:

1. **Normalizes** input data: `(x - mean) / std`
2. **Trains** on normalized data
3. **Denormalizes** final parameters back to original scale

## Class Structure

### TrainLR Class

#### Methods:

- `__init__(path)` - Initialize with dataset path and normalize data
- `calculate_coef()` - Compute parameters using normal equations
- `gradient_descent(learning_rate, iterations)` - Train using gradient descent

#### Parameters:

- `learning_rate` - Learning rate for gradient descent (default: 0.01)
- `iterations` - Number of training iterations (default: 1000)

## Example Usage

```python
from train import TrainLR

# Initialize trainer
lr = TrainLR("data.csv")

# Method 1: Normal equations (fast, analytical)
lr.calculate_coef()
theta0, theta1 = lr.theta0, lr.theta1

# Method 2: Gradient descent (iterative)
theta0, theta1 = lr.gradient_descent(learning_rate=0.01, iterations=1000)

print(f"Parameters: θ₀={theta0:.6f}, θ₁={theta1:.6f}")
```

## Output Files

### weights.txt

Contains the trained parameters in the format:
```
8484.761234
-0.021234
```

Line 1: θ₀ (intercept)
Line 2: θ₁ (slope)
