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


arr0 = [29, 78, 18, 54, 89, 80, 35, 86, 41, 70, 89, 3, 55, 84, 9, 49, 41, 2, 47, 26, 72, 10, 65, 24, 18, 46, 24, 44, 93, 74, 18, 79, 79, 35, 36, 67, 21, 5, 2, 94, 62, 50, 26, 21, 56, 14, 36, 55, 70, 73, 76, 7, 45, 38, 6, 92, 34, 74, 89, 47, 27, 29, 54, 71, 31, 83, 19, 29, 81, 86, 56, 45, 47, 52, 42, 60, 72, 12, 9, 61, 17, 69, 69, 27, 58, 79, 35, 95, 7, 8, 41, 54, 31, 14, 59, 89, 15, 27, 59, 22]
arr1 = [2300, 1900, 3600, 4100, 6900, 800, 6700, 4000, 2700, 2000, 5400, 7500, 7500, 4200, 3300, 9200, 2200, 1300, 7600, 1200, 100, 9400, 300, 5500, 8700, 2400, 4900, 500, 6900, 8500, 9900, 6500, 5600, 5500, 7700, 5700, 9400, 6500, 5000, 7600, 700, 9200, 6700, 700, 9500, 4200, 7100, 4000, 1100, 800, 3800, 1600, 7700, 4200, 2200, 4900, 2400, 9000, 000, 000, 8900, 700, 2100, 1600, 3900, 4400, 5200, 8000, 100, 9600, 7200, 800, 4200, 500, 4300, 8700, 3000, 3600, 200, 3100, 500, 4600, 3400, 8300, 1700, 100, 100, 2700, 1800, 4000, 2300, 9500, 5800, 7800, 6900, 500, 1400, 8300, 000, 2100]
x_data_3d = np.array([arr0,arr1], dtype="float64")
x_data_2d = np.array([arr0], dtype="float64")


examples = [
  # ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, 0.001, 0),
  # ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, 0.003, 0),
  # ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, 0.01, 0),
  # ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, 0.03, 0),
  # ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, 0.1, 0),

  # ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, 0.001, 0),
  # ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, 0.003, 0),
  # ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, 0.01, 0),
  # ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, 0.03, 0),
  # ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, 0.1, 0),

  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 0.003, 0),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 0.01, 0),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 0.03, 0),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 0.1, 0),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 1, 0),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 3, 0),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 10, 0),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 30, 0),


  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 0.003, 1),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 0.01, 1),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 0.03, 1),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 0.1, 1),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 1, 1),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 3, 1),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 10, 1),
  # ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, 30, 1),

  ([0,0,0,0], [-2,300000,-4, -1], np.vstack((x_data_3d, x_data_3d[1]**2)), hypothesis_logistic, calc_cost_logistic, 0.01, 0),
  ([0,0,0,0], [-2,300000,-4, -1], np.vstack((x_data_3d, x_data_3d[1]**2)), hypothesis_logistic, calc_cost_logistic, 0.1, 0),
  ([0,0,0,0], [-2,300000,-4, -1], np.vstack((x_data_3d, x_data_3d[1]**2)), hypothesis_logistic, calc_cost_logistic, 1, 0),
  ([0,0,0,0], [-2,300000,-4, -1], np.vstack((x_data_3d, x_data_3d[1]**2)), hypothesis_logistic, calc_cost_logistic, 3, 0),
]


def run():
  performance = []
  for init_thetas, real_thetas, X_ORIG, hypothesis, calc_cost, lrn_rt, reg_p in examples:
    # don't overwrite X
    X = X_ORIG.copy()
    # best practice: parameters are usually input as a column, Note: in numpy, transposing a vector makes no difference
    init_thetas = np.array(init_thetas).T
    real_thetas = np.array(real_thetas).T
    # add bias feature
    X = np.vstack((np.ones(len(X[0])), X))
    # get targets
    targets = hypothesis(X, real_thetas)

    # check that the plot function works for those thetas
    # plot_logistic_2d(X, targets, real_thetas)
    # plot_linear_2d(X, targets, real_thetas)

    X, means, sds = scale_features(X)
    start = time.time()
    res = descent(X, targets, init_thetas, hypothesis, calc_cost, lrn_rt=lrn_rt, reg_p=reg_p)

    # test model based on percent error (currently based on all data (no training-testing split))
    samples = np.array([(X_ORIG[i]-means[i])/sds[i] for i in range(len(X_ORIG))])
    samples = np.vstack((np.ones(len(samples[0])), samples))
    predictions = hypothesis(samples, res['thetas'])
    if hypothesis == hypothesis_logistic:
      percent_error = sum([1 if abs(predictions[i] - targets[i]) >= 0.5 else 0 for i in range(len(predictions))]) / len(predictions) * 100
    else:
      percent_error = max_percent_error = max(abs(predictions-targets)/targets * 100)
  
    # generate msg for performance log
    time_elapsed = f"ms={round(time.time() - start, 2)}"
    thetas = ",".join([f"p{i}={round(p,2)}" for i, p in enumerate(res['thetas'])])
    cost = f"cost={round(res['final_cost'], 2)}"
    iterations = f"its={res['iterations']}"
    msg = ";".join([thetas, cost, iterations, time_elapsed, f"pe={percent_error}"])

    performance.append(msg)

  pprint(performance)


run()
