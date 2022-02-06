import numpy as np
from plots import plot_linear_2d

from plots import plot_convergence, plot_logistic_2d

DEF_TERMINAL_COST = 0.0001
DEF_MAX_ITERATIONS = 10 ** 4
DEF_LEARNING_RATE = 0.001
DEF_REG_P = 0
DEF_MIN_DELTA = 0.001

def descent(X, targets, thetas, hypothesis, calc_cost, **kwargs):
  """Gradient descent"""
  MAX_ITS = kwargs.pop('max_its', DEF_MAX_ITERATIONS)
  LRN_RT = kwargs.pop('lrn_rt', DEF_LEARNING_RATE)
  REG_P = kwargs.pop('reg_p', DEF_REG_P)
  MIN_DELTA = kwargs.pop('min_delta', DEF_MIN_DELTA)

  m = len(targets)
  cost_history = np.array([])
  cost = float('inf')
  deltas = np.full(len(thetas), float('inf'))
  n_its = 0
  while (
      abs(sum(deltas)) > MIN_DELTA and
      n_its < MAX_ITS 
    ):
    predictions = hypothesis(thetas.T.dot(X))
    errors = predictions - targets
    deltas = (LRN_RT / m) * X.dot(errors)
    reg_subtraction = np.full(len(thetas), LRN_RT * REG_P / m)
    reg_subtraction[0] = 0
    thetas = thetas * (1-reg_subtraction) - deltas
    cost = calc_cost(m, ers=errors, ts=targets, ps=predictions, thetas=thetas, reg_p=REG_P)
    cost_history = np.append(cost_history, cost)
    n_its += 1

  plot_linear_2d(X, targets, thetas)
  # plot_logistic_2d(X[1], X[2], targets, thetas)
  # plot_convergence(cost_history)
  return [
    *[f"p{i}={round(p,2)}" for i, p in enumerate(thetas)],
    f"cost={round(cost, 2)}", 
    n_its,
    # cost_history
  ]
