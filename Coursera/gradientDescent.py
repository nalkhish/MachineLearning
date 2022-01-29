"""Playing around with gradient descent in a linear regression.

    x is of form = [
        x0, x1, x2, ...
        x0, x1, x2, ...
        ...
    ]
    params is of form = [theta0, theta1, theta2...]
"""
import time
import random
import numpy as np
import numpy.typing as npt
from typing import Callable
from pprint import pprint


DEF_TERMINAL_COST = 0.00001
DEF_MIN_PARAM_CHANGE = 0.0001
DEF_MAX_ITERATIONS = 10 ** 9
DEF_DELTA_PARAM = 0.001
DEF_LEARNING_RATE = 0.001


np_a = npt.NDArray[np.float64]
hf_type =  Callable[[np_a, np_a], np_a]
cf_type =  Callable[[np_a, np_a, np_a], float]

def linear_hypothesis(x: np_a, params: np_a) -> np_a:
    """Calculates the Y values for an array
    Example:
    x = [
        (1, 2, 0),
        (1, 1, 3),
        (1, 2, 4)
    ]
    params = [5, 2, 3]
    returns = [
        1*5 + 2*2 + 0*3    = 9,
        1*5 + 1*2 + 3*3    = 16,
        1*5 + 2*2 + 4*3    = 17
    ]
    """
    return np.matmul(x, params.reshape(params.size, 1)) 


def cost_func_linear(x: np_a, y: np_a, params: np_a):
    """Standard cost function for a linear regression
    * Uses squared differences
    """
    return ((linear_hypothesis(x, params) - y)**2).sum()/(2*len(y))



def get_delta_ys(hypothesis_func, data, cur_params):
    """get delta ys, as a row, from hypothesis and generally-formated data
    """
    return (hypothesis_func(data['x'], cur_params) - data['y']).transpose()[0]


def get_sigmas(delta_ys, X):
    """Gets sigmas, but without the division by m
    
    """
    return np.matmul(delta_ys, X)

def get_changes(LRN_RT, data, hypothesis_func, cur_params):
    """get the vector delta for gradient descent

    get_changes(
        0.01,
        {"x": [(1,2,3), (1,3,1), (1,2,2)], "y": [22, 14, 17]}, # y = [3+(2*2)+(5*3), 3+(2*3)+(5*1), 3+(2*2)+(5*2)]
        linear_hypothesis,
        [0,0,0]
    )           [-22, -14, -17] [
                                  (1, 2, 3),
       == -0.01/3 *            *  (1, 3, 1), 
                                  (1, 2, 2)
                                ]

                        
        == -0.01/3  * [-22-14-17, -28-42-28, -66-14-34]
        == -0.01/3  * [-53, -120, -114]
    """
    return (-LRN_RT/len(data['y'])) * get_sigmas(get_delta_ys(hypothesis_func, data, cur_params), data['x'])


def get_changes_unvectorized(cur_params, calc_cost, cost, data, LRN_RT):
    changes = [1.0] * len(cur_params)
    D_PARAM = 0.00001
    # for each parameter, calculate the upcoming partial derivative and change
    for i in range(len(cur_params)):
        next_params = [
            param + D_PARAM if i == j else param 
            for j, param in enumerate(cur_params)
        ]
        delta_cost = calc_cost(data['x'], data['y'], np.array(next_params))  - cost
        partial_deriv = delta_cost/D_PARAM
        changes[i] = -LRN_RT * partial_deriv
    return changes


def descent(data, calc_cost: cf_type, hypothesis_func: hf_type, initial_params: np_a, **kwargs):
    """General gradient descent"""
    T_COST = kwargs.pop('t_cost', DEF_TERMINAL_COST)
    MIN_CHG = kwargs.pop('min_chg', DEF_MIN_PARAM_CHANGE)
    MAX_ITS = kwargs.pop('max_its', DEF_MAX_ITERATIONS)
    LRN_RT = kwargs.pop('lrn_rt', DEF_LEARNING_RATE)

    cur_params = initial_params
    cost = float('inf')
    n_its = 0
    changes = [1.0] * len(cur_params)
    # keep going until (1) near local min or (2) there's been too many iterations
    while (
            # any(abs(chg) > MIN_CHG*LRN_RT for chg in changes) and
            cost > T_COST and
            n_its < MAX_ITS 
        ):
        n_its += 1
        cost = calc_cost(data['x'], data['y'], cur_params)    
        changes = get_changes(LRN_RT, data, hypothesis_func, cur_params)
        changes_unvectorized = get_changes_unvectorized(cur_params, calc_cost, cost, data, LRN_RT)
        cur_params = cur_params + changes
    return [
        *[f"p{i}={round(p,2)}" for i, p in enumerate(cur_params)],
        f"cost={round(cost, 2)}", 
        n_its
    ]


def linear_noisy_factory(noise_sd):
    """Produce linear generators with specified noise levels"""

    def linear_noisy(x: np_a, params):
        """Produce linearly-x-dependent y data, with noise"""
        return linear_hypothesis(x, params) + np.array([[random.gauss(0, noise_sd)] for _ in range(len(x))])

    return linear_noisy





x_data = np.array(list(zip([1] * 100, range(0, 100), range(100, 200))), dtype="float64")
# the data below is for one slope
x_data_linear = np.array(list(zip([1] * 100, range(0, 100))), dtype="float64")


examples = [
    # Different learning rates
    # ([0,0,0], [6,7,3], linear_hypothesis, x_data, cost_func_linear, linear_hypothesis, 0.00001),
    # ([0,0], [6,7], linear_hypothesis, x_data_linear, cost_func_linear, linear_hypothesis, 0.03),
    # ([0,0,0], [0,5,5], linear_hypothesis, x_data, cost_func_linear, linear_hypothesis, 0.0001),
    # ([100,0,0], [100,5,5], linear_hypothesis, x_data, cost_func_linear, linear_hypothesis, 0.0001),
    ([0,0,0], [6,7,3], linear_hypothesis, x_data, cost_func_linear, linear_hypothesis, 0.001),
    # ([0,0,0], [6,7,3], linear_hypothesis, x_data, cost_func_linear, linear_hypothesis, 0.01),
    # ([0,0,0], [6,7,3], linear_hypothesis, x_data, cost_func_linear, linear_hypothesis, 0.02),
    # ([0,0,0], [6,7,3], linear_hypothesis, x_data, cost_func_linear, linear_hypothesis, 0.1),

    # # Different noise levels
    # ([0,0,0], [6,7,3], linear_noisy_factory(1), x_data, cost_func_linear, linear_hypothesis, 0.001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(2), x_data, cost_func_linear, linear_hypothesis, 0.001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(3), x_data, cost_func_linear, linear_hypothesis, 0.001),

    # # Different learning rates for different noise levels
    # # noise sd = 1
    # ([0,0,0], [6,7,3], linear_noisy_factory(1), x_data, cost_func_linear, linear_hypothesis, 0.00001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(1), x_data, cost_func_linear, linear_hypothesis, 0.0001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(1), x_data, cost_func_linear, linear_hypothesis, 0.001),
    # # noise sd = 3
    # ([0,0,0], [6,7,3], linear_noisy_factory(3), x_data, cost_func_linear, linear_hypothesis, 0.00001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(3), x_data, cost_func_linear, linear_hypothesis, 0.0001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(3), x_data, cost_func_linear, linear_hypothesis, 0.001),
    # # noise sd = 5
    # ([0,0,0], [6,7,3], linear_noisy_factory(5), x_data, cost_func_linear, linear_hypothesis, 0.00001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(5), x_data, cost_func_linear, linear_hypothesis, 0.0001),
    # ([0,0,0], [6,7,3], linear_noisy_factory(5), x_data, cost_func_linear, linear_hypothesis, 0.001)
]

def scale_features(x_data):
    """
    x_data = [
        (1, 101, 2),
        (1, 99, 5),
        (1, 22, 4)
    ]

    returns =
    [
        (1, (101 - 74)/79, (2-3.67)/3),
        (1, (99 - 74)/69, (5-3.67)/3),
        (1, (22 - 74)/69, (4-3.67)/3),
    ]
    =
    [
        (1, 0.3418, -0.56)
        (1, 0.316, 0.44),
        (1, -0.659, 0.11)
    ]
    """
    n_features = len(x_data[0])
    for i in range(1, n_features):
        x_data[:, i] = (x_data[:, i] - np.mean(x_data[:,i])) / (np.max(x_data[:, i]) - np.min(x_data[:, i]))
    return x_data


def test():
    assert \
        all(linear_hypothesis(np.array([(1,2,0), (1,1,3), (1,2,4)]), np.array([5,2,3])) == np.array([[9], [16], [21]])), \
        "Something's wrong with the linear hypothesis"
    assert \
        all(get_delta_ys(linear_hypothesis, {"x": np.array([(1,2,3), (1,3,1), (1,2,2)]), "y": np.array([[22], [14], [17]])}, np.array([0, 0, 0])) == \
        [-22, -14, -17]), \
        "Something's wrong with the delta ys"
    assert \
        all(get_delta_ys(linear_hypothesis, {"x": np.array([(1,2,3), (1,3,1), (1,2,2)]), "y": np.array([[22], [14], [17]])}, np.array([1, 1, 3])) == \
        [-10, -7, -8]), \
        "Something's wrong with the delta ys"
    assert \
        all(get_changes(0.01, {"x": np.array([(1,2,3), (1,3,1), (1,2,2)]), "y": np.array([[22], [14], [17]])}, linear_hypothesis, np.array([0,0,0])) == \
            -0.01/3 * np.array([-53, -120, -114])
        ), \
        "Something's wrong with get changes"
    assert \
        all((np.round_(scale_features(np.array([(1,101,2), (1,99,5), (1,22,4)], dtype="float64")), 2) == \
            np.array([(1, 0.34, -0.56), (1, 0.32, 0.44), (1, -0.66, 0.11)])).reshape(-1)
        ), \
        "Something's wrong with feature scaling"

def run():
    test()

    performance = []
    for init_p, real_p, y_gen, x_data, cf, hf, lrn_rt in examples:
        data = {"x": x_data, "y": y_gen(x_data, np.array(real_p),)}
        x_data = scale_features(x_data)
        start = time.time()
        res = descent(data, cf, hf, np.array(init_p), lrn_rt=lrn_rt)
        time_elapsed = time.time() - start
        performance.append((res, round(time_elapsed,2)))
    pprint(performance)

run()