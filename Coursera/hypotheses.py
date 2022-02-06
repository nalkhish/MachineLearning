import numpy as np


def hypothesis_linear(X, thetas):
  return thetas.T.dot(X)


def hypothesis_logistic(X, thetas):
  return 1 / (1 + np.exp(-hypothesis_linear(X, thetas)))


def noisy_hypothesis_factory(hypothesis, noise_sd):
  """Produce noisy hypotheses"""

  def noisy_hypothesis(X, thetas):
    """Produce data with noise"""
    predictions = hypothesis(X, thetas)
    return predictions + np.random.default_rng().normal(0, noise_sd, len(predictions))

  return noisy_hypothesis
