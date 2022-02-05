import numpy as np


def scale_features(x_data):
  """Scale features to even out descent"""
  n_features = len(x_data)
  for i in range(1, n_features):
      x_data[i] = (x_data[i] - np.mean(x_data[i])) / np.std(x_data[i])
  return x_data
