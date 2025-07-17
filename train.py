import numpy as np
import math
import pandas as pd
import sys


class TrainLR:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path)
        self.X = self.data.drop(columns="price")
        self.y = self.data.drop(columns="km")  
        self.theta0 = 0
        self.theta1 = 0
        self.X_normalized = self.X.values.flatten()
        self.y_normalized = self.y.values.flatten()
        self.X_mean = np.mean(self.X_normalized)
        self.X_std = np.std(self.X_normalized)
        self.y_mean = np.mean(self.y_normalized)
        self.y_std = np.std(self.y_normalized)
        self.X_normalized = (self.X_normalized - self.X_mean) / self.X_std
        self.y_normalized = (self.y_normalized - self.y_mean) / self.y_std
        
    def calculate_coef(self):
        n = len(self.X)
        x = self.X.values.flatten()
        y = self.y.values.flatten()
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        denominator = n * sum_x2 - sum_x ** 2
        self.theta1 = (n * sum_xy - sum_x * sum_y) / denominator
        self.theta0 = (sum_y - self.theta1 * sum_x) / n
        
    def gradient_descent(self, learning_rate=0.01, iterations=1000):
        x = self.X_normalized
        y = self.y_normalized
        m = len(x)
        theta0_norm = 0
        theta1_norm = 0
        for i in range(iterations):
            predictions = theta0_norm + theta1_norm * x
            errors = predictions - y
            tmp_theta0 = learning_rate * (1/m) * np.sum(errors)
            tmp_theta1 = learning_rate * (1/m) * np.sum(errors * x)
            theta0_norm -= tmp_theta0
            theta1_norm -= tmp_theta1
            if i % 100 == 0:
                cost = np.mean(errors ** 2)
                print(f"Iteration {i}: Cost = {cost:.6f}")
        self.theta1 = theta1_norm * (self.y_std / self.X_std)
        self.theta0 = self.y_mean - self.theta1 * self.X_mean
        
        return self.theta0, self.theta1


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_file>")
        return 0
    path = sys.argv[1]
    if path == "" or path == None:
        return 0
        
    lr = TrainLR(path)
    
    # You can choose method:
    # Method 1: Normal equations (analytical)
    # lr.calculate_coef()
    # t0, t1 = lr.theta0, lr.theta1
    
    # Method 2: Gradient descent
    t0, t1 = lr.gradient_descent(learning_rate=0.01, iterations=1000)
    
    print(f"Theta 0: {t0:.6f}, Theta 1: {t1:.6f}")
    with open("weights.txt", "w") as file:
        file.write(f"{t0}\n{t1}")

if __name__ == '__main__':
    main()
    
