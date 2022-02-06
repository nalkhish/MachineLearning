import numpy as np
import matplotlib.pyplot as plt
from hypotheses import hypothesis_linear

def plot_convergence(cost_history):
  its = len(cost_history)
  if its > 100:
    plt.plot(cost_history[::its//100])
    plt.xlabel("% Iterations")
  else:
    plt.plot(cost_history)
    plt.xlabel("Iterations")
  plt.ylabel("Cost")


def plot_linear_2d(X, y, thetas):
  """Plots a scatter of + prediction line

  Args:
    x (List): predictive features
    y (List): values
    theta (List): slope parameters for the zeroth and second axis
  """
  m = len(y)
  # plot the scatter
  fig, ax = plt.subplots()
  colors = ['k'] * m
  ax.scatter(X[1], y, c=colors)
  # plot prediction line
  x_min = min(X[1])
  x_max = max(X[1])
  X_S = np.array([np.ones(m), np.linspace(x_min, x_max, m)])
  predictions = hypothesis_linear(thetas.T.dot(X_S))
  ax.plot(X_S[1], predictions, linewidth=2.0)
  plt.show()
  plt.xlabel('x')
  plt.ylabel('y')


def plot_logistic_2d(x0, x1, targets, theta):
  """Plots a scatter of boolean predictions + boundary line

  Args:
    x0 (List): elements of the first axis
    x1 (List): elements of the second axis
    targets (List): values ranging from 0 to 1, representing the probability of a True
    theta (List): for boundary line: slope parameters for the zeroth, first, and second axis
  """
  # plot the scatter, which blues representing false
  fig, ax = plt.subplots()
  colors = ['b' if p < 0.5 else 'r' for p in targets]
  ax.scatter(x0, x1, c=colors)
  # plot boundary line by finding plot's transition points (between thetaX >=0 and thetaX < 0)
  x0_max = int(max(x0))
  x0_min = int(min(x0))
  x1_max = int(max(x1))
  x1_min = int(min(x1))
  x0_b = []
  x1_b = []
  b_points = 0
  for j in range(x0_min, x0_max + 1):
    prev_prediction_is_positive = theta.dot([1, j, x1_min, x1_min**2]) >= 0
    for i in range(x1_min, x1_max + 1):
      cur_prediction_is_positive = theta.dot([1, j, i, i**2]) >= 0
      if cur_prediction_is_positive != prev_prediction_is_positive:
        x0_b.append(j)
        x1_b.append(i)
        b_points += 1
      prev_prediction_is_positive = cur_prediction_is_positive
  colors = ['k'] * b_points
  ax.scatter(x0_b, x1_b, c=colors)

  plt.show()
  plt.xlabel('x0')
  plt.ylabel('x1')
