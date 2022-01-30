"""Playing around with gradient descent in a linear regression.

    x is of form = [
        x0, x1, x2, ...
        x0, x1, x2, ...
        ...
    ]
    params is of form = [theta0, theta1, theta2...]
"""
import time
import random
import numpy as np
from pprint import pprint

DEF_TERMINAL_COST = 0.0001
DEF_MAX_ITERATIONS = 10 ** 5
DEF_LEARNING_RATE = 0.001


def scale_features(x_data):
    """Scale features to even out descent
    x_data = [
        (101, 2),
        (99, 5),
        (22, 4)
    ]

    returns =
    [
        ((101 - 74)/79, (2-3.67)/3),
        ((99 - 74)/69, (5-3.67)/3),
        ((22 - 74)/69, (4-3.67)/3),
    ]
    =
    [
        (0.3418, -0.56)
        (0.316, 0.44),
        (-0.659, 0.11)
    ]
    """
    n_features = len(x_data[0])
    for i in range(0, n_features):
        x_data[:, i] = (x_data[:, i] - np.mean(x_data[:,i])) / (np.max(x_data[:, i]) - np.min(x_data[:, i]))
    return x_data


def descent(X, targets, theta, **kwargs):
    """Gradient descent"""
    T_COST = kwargs.pop('t_cost', DEF_TERMINAL_COST)
    MAX_ITS = kwargs.pop('max_its', DEF_MAX_ITERATIONS)
    LRN_RT = kwargs.pop('lrn_rt', DEF_LEARNING_RATE)

    m = len(targets)
    cost_history = np.array([])
    cost = float('inf')
    n_its = 0
    while (
            cost > T_COST and
            n_its < MAX_ITS 
        ):
        predictions = X.dot(theta.T)
        errors = predictions - targets
        delta = (LRN_RT / m) * errors.dot(X)
        theta = theta - delta
        cost = errors.T.dot(errors)/(2*m)
        cost_history = np.append(cost_history, cost)
        n_its += 1

    return [
        *[f"p{i}={round(p,2)}" for i, p in enumerate(theta)],
        f"cost={round(cost, 2)}", 
        n_its,
        cost_history
    ]


def linear_noisy_factory(noise_sd):
    """Produce linear generators with specified noise levels"""

    def linear_noisy(X, params):
        """Produce linearly-x-dependent y data, with noise"""
        return X.dot(np.array(params).T) + np.array([[random.gauss(0, noise_sd)] for _ in range(len(X))])

    return linear_noisy


def test():
    assert \
        all((np.round_(scale_features(np.array([(101,2), (99,5), (22,4)], dtype="float64")), 2) == \
            np.array([(0.34, -0.56), (0.32, 0.44), (-0.66, 0.11)])).reshape(-1)
        ), \
        "Something's wrong with feature scaling"


x_data = np.array(list(zip(
  [29, 78, 18, 54, 89, 80, 35, 86, 41, 70, 89, 3, 55, 84, 9, 49, 41, 2, 47, 26, 72, 10, 65, 24, 18, 46, 24, 44, 93, 74, 18, 79, 79, 35, 36, 67, 21, 5, 2, 94, 62, 50, 26, 21, 56, 14, 36, 55, 70, 73, 76, 7, 45, 38, 6, 92, 34, 74, 89, 47, 27, 29, 54, 71, 31, 83, 19, 29, 81, 86, 56, 45, 47, 52, 42, 60, 72, 12, 9, 61, 17, 69, 69, 27, 58, 79, 35, 95, 7, 8, 41, 54, 31, 14, 59, 89, 15, 27, 59, 22],
  [23, 19, 36, 41, 69, 8, 67, 40, 27, 20, 54, 75, 75, 42, 33, 92, 22, 13, 76, 12, 1, 94, 3, 55, 87, 24, 49, 5, 69, 85, 99, 65, 56, 55, 77, 57, 94, 65, 50, 76, 7, 92, 67, 7, 95, 42, 71, 40, 11, 8, 38, 16, 77, 42, 22, 49, 24, 90, 0, 0, 89, 7, 21, 16, 39, 44, 52, 80, 1, 96, 72, 8, 42, 5, 43, 87, 30, 36, 2, 31, 5, 46, 34, 83, 17, 1, 1, 27, 18, 40, 23, 95, 58, 78, 69, 5, 14, 83, 0, 21],
  )),dtype="float64"
)

examples = [
    ([0,0,0], [6,7,3], x_data, 0.0001),
    ([0,0,0], [6,7,3], x_data, 0.0003),
    ([0,0,0], [6,7,3], x_data, 0.001),
    ([0,0,0], [6,7,3], x_data, 0.003),
    ([0,0,0], [6,7,3], x_data, 0.01),
    ([0,0,0], [6,7,3], x_data, 0.03),
    ([0,0,0], [6,7,3], x_data, 0.1),

    # ([0,0,0], [6,7,3], linear_noisy_factory(2), x_data, cost_func_linear, linear_hypothesis, 0.0001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(2), x_data, cost_func_linear, linear_hypothesis, 0.0003),
    # ([0,0,0], [6,7,3], linear_noisy_factory(2), x_data, cost_func_linear, linear_hypothesis, 0.001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(2), x_data, cost_func_linear, linear_hypothesis, 0.003),
    # ([0,0,0], [6,7,3], linear_noisy_factory(2), x_data, cost_func_linear, linear_hypothesis, 0.01),
    # ([0,0,0], [6,7,3], linear_noisy_factory(2), x_data, cost_func_linear, linear_hypothesis, 0.03),
    # ([0,0,0], [6,7,3], linear_noisy_factory(2), x_data, cost_func_linear, linear_hypothesis, 0.1),
]

def run():
    test()

    performance = []
    for init_p, real_p, X, lrn_rt in examples:
        X = X.copy()
        X = scale_features(X)
        X = np.hstack((np.ones((len(X),1)), X))
        targets = X.dot(np.array(real_p).T)
        start = time.time()
        res = descent(X, targets, np.array(init_p), lrn_rt=lrn_rt)
        time_elapsed = time.time() - start
        performance.append((res, round(time_elapsed,2)))
    pprint(performance)

run()