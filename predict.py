from train import Train
import numpy as np
from modifier import Modifier
import matplotlib.pyplot as plt

class Predict(Train):
    def __init__(self, data, labels):
        super().__init__(data, labels)
        self.data = data
        self.labels = labels
        self.theta = np.load('theta.npy')  # Load saved theta values.
        self.polynomial_degree = 1  # Polynomial degree (should match model's degree).
        self.sinusoid_degree = 0
        self.normalize_data = True

    def hypothesis(self, data, theta):
        """
        Hypothesis function.

        It predicts the output values y based on the input values X and model parameters.

        :param data: Data set for what the predictions will be calculated.
        :param theta: Model parameters.
        :return: Predictions made by model based on provided theta.
        """
        return data @ theta
    def predict(self, data):
        """
        Predict the output for data_set input based on trained theta values.

        :param data: Input features for prediction.
        """
        # Use only the km column for prediction (assuming km is the only feature used in the model)
        data_processed = data.iloc[:, 0].values.reshape(-1, 1)  # 'km' feature

        # Add a column of ones for the bias (intercept) term
        data_processed = np.c_[np.ones(data_processed.shape[0]), data_processed]

        # Ensure the dimensions match before prediction
        print("Data shape:", data_processed.shape)
        print("Theta shape:", self.theta.shape)

        if data_processed.shape[1] != self.theta.shape[0]:
            raise ValueError(f"Dimension mismatch: Data has {data_processed.shape[1]} features, but theta has {self.theta.shape[0]} parameters.")

        # Perform predictions
        predictions = self.hypothesis(data_processed, self.theta)
        return predictions
    def calculate_precision(self, predictions, labels):
        """
        Calculate the precision of the algorithm as the Mean Squared Error (MSE).

        :param predictions: Predicted values.
        :param labels: Actual labels.
        :return: Precision (lower is better for MSE).
        """
        mse = np.mean((predictions - labels) ** 2)
        return mse

    def plot_results(self, data, labels, predictions):
        """
        Plot the data points and the regression line.

        :param data: Input data (features).
        :param labels: Actual labels.
        :param predictions: Predictions from the model.
        """
        # Normalize data for consistency with predictions
        data_processed, *_ = super().prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )

        # Scatter original data
        plt.scatter(data.iloc[:, 0], labels, color="blue", label="Data Points")

        # Generate regression line based on normalized data
        regression_line_x = data.iloc[:, 0].sort_values()
        regression_line_x_processed, *_ = super().prepare_for_training(
            regression_line_x.to_frame(),
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )
        regression_line_y = self.hypothesis(regression_line_x_processed, self.theta)

        plt.plot(regression_line_x, regression_line_y, color="red", label="Regression Line")
        plt.xlabel("Feature 1")
        plt.ylabel("Target")
        plt.title("Linear Regression Results")
        plt.legend()
        plt.show()

def main():
    # Load data
    data_modifier = Modifier('data.csv')
    data = data_modifier.read_data()
    labels = data.iloc[:, -1]

    # Initialize Predict object
    pred = Predict(data, labels)

    # Perform predictions
    predictions = pred.predict(data)

    # Calculate and print precision
    mse = pred.calculate_precision(predictions, labels)
    print(f"Mean Squared Error (Precision): {mse}")

    # Plot results
    pred.plot_results(data, labels, predictions)

if __name__ == '__main__':
    main()
