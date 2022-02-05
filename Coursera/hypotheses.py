import numpy as np


def hypothesis_linear(thetaTX):
  return thetaTX


def hypothesis_logistic(thetaTX):
  return 1 / (1 + np.exp(-thetaTX))


def noisy_hypothesis_factory(hypothesis, noise_sd):
  """Produce linear generators with specified noise levels"""

  def noisy_hypothesis(thetaX):
    """Produce linearly-x-dependent y data, with noise"""
    return hypothesis(thetaX) + np.random.default_rng().normal(0, noise_sd, len(thetaX))

  return noisy_hypothesis
