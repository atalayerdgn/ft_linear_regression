import pandas as pd


class Modifier:
    def __init__(self,data):
        self.data = data
    def read_data(self):
        return pd.read_csv(self.data)
    def describe_data(self):
        return self.read_data().describe()
    def head_data(self):
        return self.read_data()

        