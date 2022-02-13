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
from percentErrors import percent_error_logistic, percent_error_multiclass_logistic, percent_error_standard

from plots import plot_logistic_2d, plot_linear_2d
from utilities import scale_features
from hypotheses import hypothesis_linear, hypothesis_logistic
from costs import calc_cost_linear, calc_cost_logistic, calc_cost_multiclass_logistic
from gradientDescent import descent


def get_msg(time_elapsed, percent_error, theta, cost, iterations):
  iterations = f"its={iterations}"
  return ";".join([theta, cost, iterations, time_elapsed, f"pe={percent_error}"])


def get_thetas_cost(thetas, cost):
  thetas = ",".join([f"p{i}={round(p,2)}" for i, p in enumerate(thetas)])
  cost = f"cost={round(cost, 2)}"
  return thetas, cost

def get_thetas_costs_multiclass(thetas, costs):
  thetas = ",".join([f"p{i}={round(p,2)}" for temp in thetas.T for i, p in enumerate(temp)])
  costs = ",".join([f"cost={round(cost, 2)}" for cost in costs])
  return thetas, costs


arr0 = [29, 78, 18, 54, 89, 80, 35, 86, 41, 70, 89, 3, 55, 84, 9, 49, 41, 2, 47, 26, 72, 10, 65, 24, 18, 46, 24, 44, 93, 74, 18, 79, 79, 35, 36, 67, 21, 5, 2, 94, 62, 50, 26, 21, 56, 14, 36, 55, 70, 73, 76, 7, 45, 38, 6, 92, 34, 74, 89, 47, 27, 29, 54, 71, 31, 83, 19, 29, 81, 86, 56, 45, 47, 52, 42, 60, 72, 12, 9, 61, 17, 69, 69, 27, 58, 79, 35, 95, 7, 8, 41, 54, 31, 14, 59, 89, 15, 27, 59, 22]
arr1 = [2300, 1900, 3600, 4100, 6900, 800, 6700, 4000, 2700, 2000, 5400, 7500, 7500, 4200, 3300, 9200, 2200, 1300, 7600, 1200, 100, 9400, 300, 5500, 8700, 2400, 4900, 500, 6900, 8500, 9900, 6500, 5600, 5500, 7700, 5700, 9400, 6500, 5000, 7600, 700, 9200, 6700, 700, 9500, 4200, 7100, 4000, 1100, 800, 3800, 1600, 7700, 4200, 2200, 4900, 2400, 9000, 000, 000, 8900, 700, 2100, 1600, 3900, 4400, 5200, 8000, 100, 9600, 7200, 800, 4200, 500, 4300, 8700, 3000, 3600, 200, 3100, 500, 4600, 3400, 8300, 1700, 100, 100, 2700, 1800, 4000, 2300, 9500, 5800, 7800, 6900, 500, 1400, 8300, 000, 2100]

# arr_neg_0 = [random.randint(-100, 100) for _ in range(100)]
arr_neg_0 = [29, -98, -92, -100, 62, -71, -73, -57, 53, 77, -73, 13, -62, 80, -61, -99, 23, 73, 57, 1, 100, -32, 3, 32, -38, 88, 1, -85, -57, 47, -93, -7, 85, 79, -26, 53, -61, -14, -66, 52, -67, -6, 91, -42, 33, -42, 95, 57, -3, -3, -63, -95, -5, -8, 52, -77, -43, 59, 23, -91, -75, 42, -57, 28, -51, 29, -99, 18, 34, 59, -22, -62, 92, -66, 17, 70, 91, -73, -19, 71, 49, -84, -73, 4, -3, -76, 78, -93, 38, 31, -89, -22, -36, -60, -88, 36, 15, 18, 44, -53]

x_data_3d = np.array([arr0,arr1], dtype="float64")
x_data_2d = np.array([arr0], dtype="float64")
x_data_neg_3d = np.array([arr_neg_0, arr0], dtype="float64")


examples = [
  # Basic linear regression
  ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.001, 0, get_thetas_cost),
  ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.003, 0, get_thetas_cost),
  ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.01, 0, get_thetas_cost),
  ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.03, 0, get_thetas_cost),
  ([0,0], [6,7], x_data_2d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.1, 0, get_thetas_cost),

  # Multiple linear regression
  ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.001, 0, get_thetas_cost),
  ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.003, 0, get_thetas_cost),
  ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.01, 0, get_thetas_cost),
  ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.03, 0, get_thetas_cost),
  ([0,0,0], [6,7,23], x_data_3d, hypothesis_linear, calc_cost_linear, percent_error_standard, 0.1, 0, get_thetas_cost),

  # 2d logistic regression. uses 3d data because we need an extra dimension for the choice
  ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 0.003, 0, get_thetas_cost),
  ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 0.01, 0, get_thetas_cost),
  ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 0.03, 0, get_thetas_cost),
  ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 0.1, 0, get_thetas_cost),
  ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 1, 0, get_thetas_cost),
  ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 3, 0, get_thetas_cost),
  ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 10, 0, get_thetas_cost),
  ([0,0,0], [6,120,-2], x_data_3d, hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 30, 0, get_thetas_cost),  

  # 2d logistic regression, but nonlinear boundary
  # this produces a nonlinear boundary because it uses one feature's square. the resulting fit has a low percent error. Still, it's hard to reproduce the boundary in the resulting fit because the square gets divided by a different sd
  ([0,0,0,0], [-2,300000,-4, -1], np.vstack((x_data_3d, x_data_3d[1]**2)), hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 0.01, 0, get_thetas_cost),
  ([0,0,0,0], [-2,300000,-4, -1], np.vstack((x_data_3d, x_data_3d[1]**2)), hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 0.1, 0, get_thetas_cost),
  ([0,0,0,0], [-2,300000,-4, -1], np.vstack((x_data_3d, x_data_3d[1]**2)), hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 1, 0, get_thetas_cost),
  ([0,0,0,0], [-2,300000,-4, -1], np.vstack((x_data_3d, x_data_3d[1]**2)), hypothesis_logistic, calc_cost_logistic, percent_error_logistic, 3, 0, get_thetas_cost),


  # 2d logistic regression with multiple classes
  ([[0,0,0], [0,0,0]], [[6,-200, 2],[6,120,-2]], x_data_3d, hypothesis_logistic, calc_cost_multiclass_logistic, percent_error_multiclass_logistic, 0.003, 0, get_thetas_costs_multiclass),
]


def run():
  performance = []
  for init_thetas, real_thetas, X_ORIG, hypothesis, calc_cost, calc_pe, lrn_rt, reg_p, parse_thetas_costs in examples:
    # don't overwrite X
    X = X_ORIG.copy()
    # best practice: weight parameters usually have different features along the col axis, Note: in numpy, transposing a vector makes no difference
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
    percent_error = calc_pe(predictions, targets)
  
    # generate msg for performance log
    time_elapsed = f"ms={round(time.time() - start, 2)}"
    thetas, cost = parse_thetas_costs(res['thetas'], res['final_cost'])
    msg = get_msg(time_elapsed, percent_error, thetas, cost, res['iterations'])

    performance.append(msg)

  pprint(performance)


run()
