import numpy as np


def scale_features(x_data):
  """Scale features to even out descent"""
  means = []
  sds = []
  n_features = len(x_data)
  for i in range(1, n_features):
    mean = np.mean(x_data[i])
    means.append(mean)
    sd = np.std(x_data[i])
    sds.append(sd)
    x_data[i] = (x_data[i] - mean) / sd
  return x_data, means, sds
