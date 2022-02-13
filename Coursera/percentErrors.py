def percent_error_standard(predictions, targets):
  percent_error = max_percent_error = max(abs(predictions-targets)/targets) * 100
  return percent_error


def percent_error_logistic(predictions, targets):
  return sum([1 if abs(predictions[i] - targets[i]) >= 0.5 else 0 for i in range(len(predictions))]) / len(predictions) * 100


def percent_error_multiclass_logistic(predictions, targets):
  percent_error = 0
  for w in range(len(predictions)):
    percent_error += sum([1 if abs(predictions[w][i] - targets[w][i]) >= 0.5 else 0 for i in range(len(predictions[w]))]) / len(predictions[w]) * 100
  percent_error /= len(predictions)
  return percent_error

