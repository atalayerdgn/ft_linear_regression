import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




class Vis:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.X = self.data['km']
        self.y = self.data['price']
        self.theta0 = 0
        self.theta1 = 0
    def plot_hist(self):
        column_names = [i for i in self.data.columns]
        for i in column_names:
            plt.hist(self.data[i], bins=10)
            plt.title(f"Histogram of {i}")
            plt.xlabel(i)
            plt.ylabel('Frequency')
            plt.show()
    def plot_boxplot(self):
        plt.boxplot(self.X)
        plt.title('Boxplot of km')
        plt.xlabel('km')
        plt.ylabel('Value')
        plt.show()
    def repartition(self):
        plt.scatter(self.X, self.y)
        plt.title('Repartition of km vs price')
        plt.xlabel('km')
        plt.ylabel('price')
        plt.show()
    def plot_scatter(self):
        plt.scatter(self.X, self.y)
        with open("weights.txt") as f:
            theta0, theta1 = float(f.readline()), float(f.readline())
        x_line = np.linspace(self.X.min(), self.X.max(), 100)
        y_line = theta0 + theta1 * x_line
        plt.plot(x_line, y_line, 'r-', label='Regression Line')
        plt.title('Scatter plot of km vs price')
        plt.xlabel('km')
        plt.ylabel('price')
        plt.legend()
        plt.show()
    def calculate_confusion_matrix(self, theta0, theta1):
        predictions = theta0 + theta1 * self.X
        threshold = np.mean(self.y)
        actual_binary = (self.y > threshold).astype(int)
        pred_binary = (predictions > threshold).astype(int)
        tp = np.sum((actual_binary == 1) & (pred_binary == 1))  # True Positive
        tn = np.sum((actual_binary == 0) & (pred_binary == 0))  # True Negative  
        fp = np.sum((actual_binary == 0) & (pred_binary == 1))  # False Positive
        fn = np.sum((actual_binary == 1) & (pred_binary == 0))  # False Negative
        confusion_matrix = np.array([[tn, fp], 
                                   [fn, tp]])
        def calculate_precision(tp, fp):
            if tp + fp == 0:
                return 0.0
            return tp / (tp + fp)
        
        def calculate_recall(tp, fn):
            if tp + fn == 0:
                return 0.0
            return tp / (tp + fn)
        
        def calculate_f1_score(precision, recall):
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
        
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        f1_score = calculate_f1_score(precision, recall)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        return confusion_matrix
    
def main():
    path = 'data.csv'
    v = Vis(path)
    v.plot_hist()
    v.plot_boxplot()
    v.repartition()
    v.plot_scatter()
    with open("weights.txt") as f:
        theta0, theta1 = float(f.readline()), float(f.readline())
    v.calculate_confusion_matrix(theta0, theta1)
if __name__ == '__main__':
    main()
