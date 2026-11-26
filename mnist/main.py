import random

import keras
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(y_predicted, y_actual):
    return (y_actual - y_predicted) ** 2


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_flat = x_train[0].flatten()

    l1 = np.array(x_train[0].flatten() / 255.0)
    weighs = np.random.randn(len(l1)) * 0.01

    bias = 0.0

    output = sigmoid(np.dot(l1, weighs) + bias)

    y_actual = float(y_train[0] == 5)
    error = loss(output, y_actual)

    print(f"Output: {output}")
    print(f"Error: {error}")


if __name__ == "__main__":
    main()
