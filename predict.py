import numpy as np
import math
import pandas as pd
import sys


class Predict:
    def __init__(self):
        pass
    @staticmethod
    def predict(x):
        t0 = 0
        t1 = 0
        with open("weights.txt") as file:
            t0 = float(file.readline())
            t1 = float(file.readline())
        return t0 + (t1 * x)
def main():
    print("For Exit press 0 or press 1")
    while True:
        x = int(input("Enter : "))
        if x == 1:
            y = float(input("Enter km of Car : "))
            if y > 0:
                print(f"The price of car is {Predict.predict(y)}")
            else:
                print("İnvalid Arguments")
        elif x == 0:
            break
        
if __name__ == '__main__':
    main()
