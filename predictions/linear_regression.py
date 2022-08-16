from methods.linear_regression import linear_regression as method

import numpy as np

""" for our current problem the model is:
`y = theta_0 + theta_1 * x` """


def __model(x, theta):
    return np.matmul(x, theta)


def __cost_function(x, y, theta):
    return np.sum((y - __model(x, theta)) ** 2)


def __gradient_descent(x, y, theta, learning_rate=0.1, num_epochs=10):
    m = x.shape[0]
    all_costs = []

    for _ in range(num_epochs):
        f_x = __model(x, theta)
        cost_ = (1 / m) * (x.T @ (f_x - y))
        theta = theta - learning_rate * cost_
        all_costs.append(__cost_function(x, y, theta))

    return (theta, all_costs)


def __data_length(data):
    return len(data)


__theta = np.zeros((__data_length, 1))
__learning_rate = 0.1
__num_epochs = 50


def __train(x, y):
    return __gradient_descent(x, y, __theta, __learning_rate, __num_epochs)


def __predict(x, theta):
    return __model(x, theta)


linear_regression = method(__train, __predict)
