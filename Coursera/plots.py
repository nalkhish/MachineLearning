import matplotlib.pyplot as plt


def plot_convergence(cost_history):
    its = len(cost_history)
    plt.plot(cost_history[::its//100])
    plt.ylabel("Cost")
    plt.xlabel("% Iterations")
    r = ''


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
