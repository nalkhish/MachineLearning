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

from plots import plot_logistic_2d, plot_linear_2d
from utilities import scale_features
from hypotheses import hypothesis_linear, hypothesis_logistic
from costs import calc_cost_linear, calc_cost_logistic
from gradientDescent import descent


x_data = np.array(
  [
    [29, 78, 18, 54, 89, 80, 35, 86, 41, 70, 89, 3, 55, 84, 9, 49, 41, 2, 47, 26, 72, 10, 65, 24, 18, 46, 24, 44, 93, 74, 18, 79, 79, 35, 36, 67, 21, 5, 2, 94, 62, 50, 26, 21, 56, 14, 36, 55, 70, 73, 76, 7, 45, 38, 6, 92, 34, 74, 89, 47, 27, 29, 54, 71, 31, 83, 19, 29, 81, 86, 56, 45, 47, 52, 42, 60, 72, 12, 9, 61, 17, 69, 69, 27, 58, 79, 35, 95, 7, 8, 41, 54, 31, 14, 59, 89, 15, 27, 59, 22],
    # [2300, 1900, 3600, 4100, 6900, 800, 6700, 4000, 2700, 2000, 5400, 7500, 7500, 4200, 3300, 9200, 2200, 1300, 7600, 1200, 100, 9400, 300, 5500, 8700, 2400, 4900, 500, 6900, 8500, 9900, 6500, 5600, 5500, 7700, 5700, 9400, 6500, 5000, 7600, 700, 9200, 6700, 700, 9500, 4200, 7100, 4000, 1100, 800, 3800, 1600, 7700, 4200, 2200, 4900, 2400, 9000, 000, 000, 8900, 700, 2100, 1600, 3900, 4400, 5200, 8000, 100, 9600, 7200, 800, 4200, 500, 4300, 8700, 3000, 3600, 200, 3100, 500, 4600, 3400, 8300, 1700, 100, 100, 2700, 1800, 4000, 2300, 9500, 5800, 7800, 6900, 500, 1400, 8300, 000, 2100],
  ]
  , dtype="float64"
)

examples = [
  # ([0,0], [6,7], x_data, hypothesis_linear, calc_cost_linear, 0.00003, 0),
  # ([0,0], [6,7], x_data, hypothesis_linear, calc_cost_linear, 0.0001, 0),
  # ([0,0], [6,7], x_data, hypothesis_linear, calc_cost_linear, 0.0003, 0),
  ([0,0], [6,7], x_data, hypothesis_linear, calc_cost_linear, 0.001, 0),
  ([0,0], [6,7], x_data, hypothesis_linear, calc_cost_linear, 0.003, 0),
  ([0,0], [6,7], x_data, hypothesis_linear, calc_cost_linear, 0.01, 0),
  ([0,0], [6,7], x_data, hypothesis_linear, calc_cost_linear, 0.03, 0),
  ([0,0], [6,7], x_data, hypothesis_linear, calc_cost_linear, 0.1, 0),

  # ([0,0,0], [6,7,23], x_data, hypothesis_linear, calc_cost_linear, 0.00003, 0),
  # ([0,0,0], [6,7,23], x_data, hypothesis_linear, calc_cost_linear, 0.0001, 0),
  # ([0,0,0], [6,7,23], x_data, hypothesis_linear, calc_cost_linear, 0.0003, 0),
  # ([0,0,0], [6,7,23], x_data, hypothesis_linear, calc_cost_linear, 0.001, 0),
  # ([0,0,0], [6,7,23], x_data, hypothesis_linear, calc_cost_linear, 0.003, 0),
  # ([0,0,0], [6,7,23], x_data, hypothesis_linear, calc_cost_linear, 0.01, 0),
  # ([0,0,0], [6,7,23], x_data, hypothesis_linear, calc_cost_linear, 0.03, 0),
  # ([0,0,0], [6,7,23], x_data, hypothesis_linear, calc_cost_linear, 0.1, 0),

  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.00003, 10),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.0001, 10),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.0003, 10),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.001, 10),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.003, 10),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.01, 10),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.03, 10),
  # ([0,0,0], [6,7,-3], x_data, hypothesis_logistic, calc_cost_logistic, 0.1, 10),

  # ([0,0,0,0], [-2,30,-4, -1], np.vstack((x_data, x_data[1]**2)), hypothesis_logistic, calc_cost_logistic, 1),
]


def run():
  performance = []
  for init_p, real_p, X, hypothesis, calc_cost, lrn_rt, reg_p in examples:
    # best practice: parameters are usually input as a column, 
    # Note: in numpy, transposing a vector makes no difference
    init_p = np.array(init_p).T
    real_p = np.array(real_p).T
    X = X.copy()
    m = len(X[0])
    X = np.vstack((np.ones(m), X))
    targets = hypothesis(real_p.T.dot(X))

    # plot_logistic_2d(X[1], X[2], targets, real_p)
    # plot_linear_2d(X, targets, real_p)

    X, means, sds = scale_features(X)

    start = time.time()
    res = descent(X, targets, init_p, hypothesis, calc_cost, lrn_rt=lrn_rt, reg_p=reg_p)
    time_elapsed = time.time() - start
    performance.append((res, round(time_elapsed,2)))
  pprint(performance)


run()