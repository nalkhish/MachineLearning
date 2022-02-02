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
import matplotlib.pyplot as plt



DEF_TERMINAL_COST = 0.0001
DEF_MAX_ITERATIONS = 10 ** 5
DEF_LEARNING_RATE = 0.001


def plot_convergence(cost_history):
    its = len(cost_history)
    plt.plot(cost_history[::its//100])
    plt.ylabel("Cost")
    plt.xlabel("% Iterations")
    r = ''

def plot_logistic_2d(x0, x1, targets, theta):
    fig, ax = plt.subplots()
    colors = ['b' if p < 0.5 else 'r' for p in targets]
    ax.scatter(x0, x1, c=colors)
    # plot boundary line by finding plot's transition points (between thetaX >=0 and thetaX < 0)
    x0_max = int(max(x0))
    x0_min = int(min(x0))
    x1_max = int(max(x1))
    x1_min = int(min(x1))
    x0_b = []
    x1_b = []
    b_points = 0
    prev_prediction_is_positive = theta.dot([1, x0_min, x1_min]) >= 0
    for i in range(x0_min + 1, x0_max + 1):
        for j in range(x1_min - 1, x1_max + 1):
            cur_prediction_is_positive = theta.dot([1, i, j]) >= 0
            if cur_prediction_is_positive != prev_prediction_is_positive:
                x0_b.append(i)
                x1_b.append(j)
                b_points += 1
            prev_prediction_is_positive = cur_prediction_is_positive
    colors = ['k'] * b_points
    ax.scatter(x0_b, x1_b, c=colors)

    plt.show()
    plt.xlabel('x0')
    plt.ylabel('x1')


def scale_features(x_data):
    """Scale features to even out descent"""
    n_features = len(x_data)
    for i in range(1, n_features):
        x_data[i] = (x_data[i] - np.mean(x_data[i])) / (np.max(x_data[i]) - np.min(x_data[i]))
    return x_data


def calc_cost_linear(m, **kwargs):
    return kwargs['ers'].T.dot(kwargs['ers'])/(2*m)


def calc_cost_logistic(m, **kwargs):
    return (-kwargs['ts'].T.dot(np.log(kwargs['ps'])) - (1-kwargs['ts']).T.dot(np.log(1-kwargs['ps']))) / m


def hypothesis_linear(thetaTX):
    return thetaTX


def hypothesis_logistic(thetaTX):
    return 1 / (1 + np.exp(-thetaTX))


def linear_noisy_factory(noise_sd):
    """Produce linear generators with specified noise levels"""

    def linear_noisy(X, params):
        """Produce linearly-x-dependent y data, with noise"""
        return X.dot(np.array(params).T) + np.array([[random.gauss(0, noise_sd)] for _ in range(len(X))])

    return linear_noisy


def descent(X, targets, theta, hypothesis, calc_cost, **kwargs):
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
        predictions = hypothesis(theta.T.dot(X))
        errors = predictions - targets
        delta = (LRN_RT / m) * X.dot(errors)
        theta = theta - delta
        cost = calc_cost(m, ers=errors, ts=targets, ps=predictions)
        cost_history = np.append(cost_history, cost)
        n_its += 1

    plot_logistic_2d(X[1], X[2], targets, theta)
    # plot_convergence(cost_history)
    return [
        *[f"p{i}={round(p,2)}" for i, p in enumerate(theta)],
        f"cost={round(cost, 2)}", 
        n_its,
        # cost_history
    ]


x_data = np.array(
    [
        [29, 78, 18, 54, 89, 80, 35, 86, 41, 70, 89, 3, 55, 84, 9, 49, 41, 2, 47, 26, 72, 10, 65, 24, 18, 46, 24, 44, 93, 74, 18, 79, 79, 35, 36, 67, 21, 5, 2, 94, 62, 50, 26, 21, 56, 14, 36, 55, 70, 73, 76, 7, 45, 38, 6, 92, 34, 74, 89, 47, 27, 29, 54, 71, 31, 83, 19, 29, 81, 86, 56, 45, 47, 52, 42, 60, 72, 12, 9, 61, 17, 69, 69, 27, 58, 79, 35, 95, 7, 8, 41, 54, 31, 14, 59, 89, 15, 27, 59, 22],
        [23, 19, 36, 41, 69, 8, 67, 40, 27, 20, 54, 75, 75, 42, 33, 92, 22, 13, 76, 12, 1, 94, 3, 55, 87, 24, 49, 5, 69, 85, 99, 65, 56, 55, 77, 57, 94, 65, 50, 76, 7, 92, 67, 7, 95, 42, 71, 40, 11, 8, 38, 16, 77, 42, 22, 49, 24, 90, 0, 0, 89, 7, 21, 16, 39, 44, 52, 80, 1, 96, 72, 8, 42, 5, 43, 87, 30, 36, 2, 31, 5, 46, 34, 83, 17, 1, 1, 27, 18, 40, 23, 95, 58, 78, 69, 5, 14, 83, 0, 21],
    ]
    ,dtype="float64"
)

examples = [
    # ([0,0,0], [6,7,3], x_data, hypothesis_linear, calc_cost_linear, 0.00003),
    # ([0,0,0], [6,7,3], x_data, hypothesis_linear, calc_cost_linear, 0.0001),
    # ([0,0,0], [6,7,3], x_data, hypothesis_linear, calc_cost_linear, 0.0003),
    # ([0,0,0], [6,7,3], x_data, hypothesis_linear, calc_cost_linear, 0.001),
    # ([0,0,0], [6,7,3], x_data, hypothesis_linear, calc_cost_linear, 0.003),
    # ([0,0,0], [6,7,3], x_data, hypothesis_linear, calc_cost_linear, 0.01),
    # ([0,0,0], [6,7,3], x_data, hypothesis_linear, calc_cost_linear, 0.03),
    # ([0,0,0], [6,7,3], x_data, hypothesis_linear, calc_cost_linear, 0.1),

    # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.00003),
    # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.0001),
    # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.0003),
    # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.001),
    # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.003),
    ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.01),
    ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.03),
    ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.1),
]

def run():
    performance = []
    for init_p, real_p, X, hypothesis, calc_cost, lrn_rt in examples:
        # best practice: parameters are usually input as a column, 
        # Note: in numpy, transposing a vector makes no difference
        init_p = np.array(init_p).T
        real_p = np.array(real_p).T
        X = X.copy()
        m = len(X[0])
        X = np.vstack((np.ones(m), X))
        targets = hypothesis(real_p.T.dot(X))
        # X = scale_features(X)

        start = time.time()
        res = descent(X, targets, init_p, hypothesis, calc_cost, lrn_rt=lrn_rt)
        time_elapsed = time.time() - start
        performance.append((res, round(time_elapsed,2)))
    pprint(performance)

run()