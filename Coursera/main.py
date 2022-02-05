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

from plots import plot_logistic_2d
from utilities import scale_features
from hypotheses import hypothesis_linear, hypothesis_logistic
from costs import calc_cost_linear, calc_cost_logistic
from gradientDescent import descent


x_data = np.array(
  [
    [29, 78, 18, 54, 89, 80, 35, 86, 41, 70, 89, 3, 55, 84, 9, 49, 41, 2, 47, 26, 72, 10, 65, 24, 18, 46, 24, 44, 93, 74, 18, 79, 79, 35, 36, 67, 21, 5, 2, 94, 62, 50, 26, 21, 56, 14, 36, 55, 70, 73, 76, 7, 45, 38, 6, 92, 34, 74, 89, 47, 27, 29, 54, 71, 31, 83, 19, 29, 81, 86, 56, 45, 47, 52, 42, 60, 72, 12, 9, 61, 17, 69, 69, 27, 58, 79, 35, 95, 7, 8, 41, 54, 31, 14, 59, 89, 15, 27, 59, 22],
    [23, 19, 36, 41, 69, 8, 67, 40, 27, 20, 54, 75, 75, 42, 33, 92, 22, 13, 76, 12, 1, 94, 3, 55, 87, 24, 49, 5, 69, 85, 99, 65, 56, 55, 77, 57, 94, 65, 50, 76, 7, 92, 67, 7, 95, 42, 71, 40, 11, 8, 38, 16, 77, 42, 22, 49, 24, 90, 0, 0, 89, 7, 21, 16, 39, 44, 52, 80, 1, 96, 72, 8, 42, 5, 43, 87, 30, 36, 2, 31, 5, 46, 34, 83, 17, 1, 1, 27, 18, 40, 23, 95, 58, 78, 69, 5, 14, 83, 0, 21],
  ]
  , dtype="float64"
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
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.01),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.03),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.1),

  ([0,0,0,0], [-2,30,-4, -1], np.vstack((x_data, x_data[1]**2)), hypothesis_logistic, calc_cost_logistic, 1),
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

    # plot_logistic_2d(X[1], X[2], targets, real_p)

    # X = scale_features(X)

    start = time.time()
    res = descent(X, targets, init_p, hypothesis, calc_cost, lrn_rt=lrn_rt)
    time_elapsed = time.time() - start
    performance.append((res, round(time_elapsed,2)))
  pprint(performance)


run()