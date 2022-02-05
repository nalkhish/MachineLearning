import numpy as np


def calc_cost_linear(m, **kwargs):
  return kwargs['ers'].T.dot(kwargs['ers'])/(2*m)


def calc_cost_logistic(m, **kwargs):
  return (-kwargs['ts'].T.dot(np.log(kwargs['ps'])) - (1-kwargs['ts']).T.dot(np.log(1-kwargs['ps']))) / m
