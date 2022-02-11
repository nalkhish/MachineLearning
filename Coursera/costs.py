import numpy as np


def regularization_cost(kwargs):
  thetas = kwargs.get('thetas', np.array([0]))
  return kwargs.get('reg_p', 0) * thetas.T.dot(thetas)

def regularization_cost_2(thetas, kwargs):
  return kwargs.get('reg_p', 0) * thetas.T.dot(thetas)

def calc_cost_linear(m, **kwargs):
  return kwargs['ers'].T.dot(kwargs['ers'])/(2*m) + regularization_cost(kwargs)


def calc_cost_logistic(m, **kwargs):
  targets = kwargs['ts']
  predictions = kwargs['ps']
  return (-targets.T.dot(np.log(predictions)) - (1-targets).T.dot(np.log(1-predictions))) / m + regularization_cost(kwargs)


def calc_cost_multiclass_logistic(m, **kwargs):
  targets = kwargs['ts']
  predictions = kwargs['ps']
  costs = []
  for i in range(len(targets)):
    thetas = kwargs.get('thetas', np.array([0] * len(targets))).T[i]
    cost = (-targets[i].T.dot(np.log(predictions[i])) - (1-targets[i]).T.dot(np.log(1-predictions[i]))) / m + regularization_cost_2(thetas, kwargs)
    costs.append(cost)
  return costs
