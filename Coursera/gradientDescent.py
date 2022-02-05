import numpy as np

from plots import plot_convergence, plot_logistic_2d

DEF_TERMINAL_COST = 0.0001
DEF_MAX_ITERATIONS = 10 ** 5
DEF_LEARNING_RATE = 0.001


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
