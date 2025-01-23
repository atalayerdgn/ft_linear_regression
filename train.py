from modifier import Modifier
import numpy as np

class Train:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        Initialize the training class with data and labels.
        """
        (
            data_processed,
            features_mean,
            features_deviation
        ) = self.prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.labels = labels
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

    def generate_polynomials(self, dataset, polynomial_degree, normalize_data=False):
        """
        Generate polynomial features up to the given degree.
        """
        features_split = np.array_split(dataset, 2, axis=1)
        dataset_1 = features_split[0]
        dataset_2 = features_split[1]

        (num_examples_1, num_features_1) = dataset_1.shape
        (num_examples_2, num_features_2) = dataset_2.shape

        if num_examples_1 != num_examples_2:
            raise ValueError('Cannot generate polynomials for sets with different row numbers')

        if num_features_1 == 0 and num_features_2 == 0:
            raise ValueError('Cannot generate polynomials for sets with no columns')

        if num_features_1 == 0:
            dataset_1 = dataset_2
        elif num_features_2 == 0:
            dataset_2 = dataset_1

        num_features = min(num_features_1, num_features_2)
        dataset_1 = dataset_1[:, :num_features]
        dataset_2 = dataset_2[:, :num_features]

        polynomials = np.empty((num_examples_1, 0))

        for i in range(1, polynomial_degree + 1):
            for j in range(i + 1):
                polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
                polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

        if normalize_data:
            polynomials_mean = np.mean(polynomials, 0)
            polynomials_deviation = np.std(polynomials, 0)
            polynomials_deviation[polynomials_deviation == 0] = 1
            polynomials = (polynomials - polynomials_mean) / polynomials_deviation

        return polynomials

    def generate_sinusoids(self, dataset, sinusoid_degree):
        """
        Generate sinusoidal features based on the input data.
        """
        num_examples = dataset.shape[0]
        sinusoids = np.empty((num_examples, 0))

        for degree in range(1, sinusoid_degree + 1):
            sinusoid_features = np.sin(degree * dataset)
            sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

        return sinusoids

    def prepare_for_training(self, data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        Preprocess the data by normalizing and generating polynomial and sinusoidal features.
        """
        num_samples = data.shape[0]
        processed_data = np.array(data.copy())
        features_mean = 0
        features_deviation = 0
        data_norm = processed_data

        if processed_data.any():
            features_normalized = np.copy(processed_data).astype(float)
            features_mean = np.mean(processed_data, 0)
            features_deviation = np.std(processed_data, 0)
            if processed_data.shape[0] > 1:
                features_normalized -= features_mean
            features_deviation[features_deviation == 0] = 1
            features_normalized /= features_deviation
            data_norm = features_normalized

        data_processed = data_norm

        if polynomial_degree > 0:
            polynomials = self.generate_polynomials(data_norm, polynomial_degree, normalize_data)
            data_processed = np.concatenate((data_processed, polynomials), axis=1)

        if sinusoid_degree > 0:
            sinusoids = self.generate_sinusoids(data_norm, sinusoid_degree)
            data_processed = np.concatenate((data_processed, sinusoids), axis=1)

        data_processed = np.concatenate((np.ones((num_samples, 1)), data_processed), axis=1)

        return data_processed, features_mean, features_deviation

    def gradient_descent(self, alpha, lambda_, num_iters):
        """
        Perform gradient descent to learn theta.
        """
        num_samples, num_features = self.data.shape
        theta = np.zeros(num_features)
        cost_h = np.zeros(num_iters)

        for i in range(num_iters):
            h = self.data @ theta
            theta -= alpha * (1 / num_samples) * self.data.T @ (h - self.labels) + (lambda_ / num_samples) * theta
            cost_h[i] = (1 / (2 * num_samples)) * np.sum((h - self.labels) ** 2) + (lambda_ / (2 * num_samples)) * np.sum(theta ** 2)

        return theta, cost_h

    def train(self, alpha, lambda_, num_iters):
        """
        Train the model and return theta and cost history.
        """
        self.theta, cost_h = self.gradient_descent(alpha, lambda_, num_iters)
        return self.theta, cost_h

    def save_theta(self, theta):
        """
        Save the learned theta parameters to a file.
        """
        np.save('theta', theta)


def main():
    # Load the data
    data = Modifier('data.csv').read_data()
    
    # Extract features (km) and labels (price)
    km = data.iloc[:, 0].values  # km as feature
    price = data.iloc[:, 1].values  # price as label
    
    # Convert km to a 2D array (for compatibility with linear regression)
    km = km.reshape(-1, 1)
    
    # Initialize and train the model
    train = Train(km, price)
    theta, cost_h = train.train(0.01, 0.01, 1000)
    train.save_theta(theta)

if __name__ == '__main__':
    main()
