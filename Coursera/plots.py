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
  predictions = hypothesis_linear(X_S, thetas)
  ax.plot(X_S[1], predictions, linewidth=2.0)
  plt.show()
  plt.xlabel('x')
  plt.ylabel('y')


def plot_logistic_2d(X, targets, theta):
  """Plots a scatter of boolean predictions + boundary line

  Args:
    x (List): Matrix
    targets (List): values ranging from 0 to 1, representing the probability of a True
    theta (List): for boundary line: slope parameters for the zeroth, first, and second axis
  """
  BOUNDARY_RESOLUTION = 100
  x0, x1 = X[1], X[2]
  fig, ax = plt.subplots()
  # plot boundary by contouring based on prediction
  x0_max = max(x0)
  x0_min = min(x0)
  x1_max = max(x1)
  x1_min = min(x1)
  X0, X1 = np.meshgrid(np.linspace(x0_min, x0_max, BOUNDARY_RESOLUTION), np.linspace(x1_min, x1_max, BOUNDARY_RESOLUTION))

  TARGET_COLORS = ['red', 'yellow']
  TRANSPARENCY = 0.2
  for w in range(len(theta.T)):
    Z = np.zeros(X0.shape)
    for i in range(len((Z))):
      for j in range(len(Z)):
        prediction = theta.T[w].dot([1, X0[i][j], X1[i][j]])
        if prediction >= 0:
          Z[i][j] = 1
        else:
          Z[i][j] = 0
    # CS = ax.contour(X0, X1, Z, colors=['red', 'blue', 'black'][w])
    # ax.clabel(CS, inline=True, fontsize=20, manual=[(-5,0)])
    ax.contourf(X0, X1, Z, levels=[0, 0.5, 1], colors=['blue', TARGET_COLORS[w]], alpha=TRANSPARENCY)
    plt.show()


  # plot the scatter, where blues represent false.
  for w in range(len(targets)):
    colors = ['blue' if p < 0.5 else TARGET_COLORS[w] for p in targets[w]]
    ax.scatter(x0, x1, c=colors, alpha=TRANSPARENCY)


  plt.show()
  plt.xlabel('x0')
  plt.ylabel('x1')
