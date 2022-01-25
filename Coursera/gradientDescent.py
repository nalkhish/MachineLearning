"""Playing around with gradient descent in a linear regression."""
import time
import random
from typing import List
from pprint import pprint


DEF_TERMINAL_COST = 0.0001
DEF_MIN_PARAM_CHANGE = 0.0001
DEF_MAX_ITERATIONS = 10 ** 9
DEF_DELTA_PARAM = 0.001
DEF_LEARNING_RATE = 0.001


def descent(data, calc_cost, initial_params, **kwargs):
    """General gradient descent"""
    T_COST = kwargs.pop('t_cost', DEF_TERMINAL_COST)
    MIN_CHG = kwargs.pop('min_chg', DEF_MIN_PARAM_CHANGE)
    MAX_ITS = kwargs.pop('max_its', DEF_MAX_ITERATIONS)
    D_PARAM = kwargs.pop('d_param', DEF_DELTA_PARAM)
    LRN_RT = kwargs.pop('lrn_rt', DEF_LEARNING_RATE)

    cur_params = initial_params
    cost = float('inf')
    n_its = 0
    changes = [1] * len(cur_params)
    # keep going until (1) near local min or (2) there's been too many iterations
    while (
            any(abs(chg) > MIN_CHG for chg in changes) and
            cost > T_COST and 
            n_its < MAX_ITS 
        ):
        n_its += 1
        cost = calc_cost(data, cur_params)    
        # for each parameter, calculate the upcoming partial derivative and change
        for i in range(len(cur_params)):
            next_params = [
                param + D_PARAM if i == j else param 
                for j, param in enumerate(cur_params)
            ]
            delta_cost = calc_cost(data, next_params) - cost
            partial_deriv = delta_cost/D_PARAM
            changes[i] = -LRN_RT * partial_deriv
        cur_params = [cur_params[i] + changes[i] for i in range(len(cur_params))]
    return [
        *[f"p{i}={round(p,2)}" for i, p in enumerate(cur_params)],
        f"cost={round(cost, 2)}", 
        n_its
    ]


def linear(params, x_list: List[int]):
    """Produce linearly-x-dependent y data"""
    p0, p1 = params
    return [p0 + p1 * x for x in x_list]


def linear_noisy_factory(noise_sd):
    """Produce linear generators with specified noise levels"""

    def linear_noisy(params, x_list: List[int]):
        """Produce linearly-x-dependent y data, with noise"""

        p0, p1 = params
        return [p0 + p1 * x + random.gauss(0, noise_sd) for x in x_list]

    return linear_noisy


def cost_func_linear_1(data, params):
    """Standard cost function for a linear regression

    * Uses squared differences
    """
    p0, p1 = params
    return sum([((p0 + x * p1) - y_actual) ** (2) for x, y_actual in data])


x_data = list(range(0, 10))
tests = [
    # Different learning rates
    ([0,1], [7,3], linear, x_data, cost_func_linear_1, 0.00001),
    ([0,1], [7,3], linear, x_data, cost_func_linear_1, 0.0001),
    ([0,1], [7,3], linear, x_data, cost_func_linear_1, 0.001),

    # Different noise levels
    ([0,1], [7,3], linear_noisy_factory(1), x_data, cost_func_linear_1, 0.001),
    ([0,1], [7,3], linear_noisy_factory(2), x_data, cost_func_linear_1, 0.001),
    ([0,1], [7,3], linear_noisy_factory(3), x_data, cost_func_linear_1, 0.001),

    # Different learning rates for different noise levels
    # noise sd = 1
    ([0,1], [7,3], linear_noisy_factory(1), x_data, cost_func_linear_1, 0.00001),
    ([0,1], [7,3], linear_noisy_factory(1), x_data, cost_func_linear_1, 0.0001),
    ([0,1], [7,3], linear_noisy_factory(1), x_data, cost_func_linear_1, 0.001),
    # noise sd = 3
    ([0,1], [7,3], linear_noisy_factory(3), x_data, cost_func_linear_1, 0.00001),
    ([0,1], [7,3], linear_noisy_factory(3), x_data, cost_func_linear_1, 0.0001),
    ([0,1], [7,3], linear_noisy_factory(3), x_data, cost_func_linear_1, 0.001),
    # noise sd = 5
    ([0,1], [7,3], linear_noisy_factory(5), x_data, cost_func_linear_1, 0.00001),
    ([0,1], [7,3], linear_noisy_factory(5), x_data, cost_func_linear_1, 0.0001),
    ([0,1], [7,3], linear_noisy_factory(5), x_data, cost_func_linear_1, 0.001),
]


def run():
    performance = []
    for init_p, real_p, y_gen, x_data, cost_func, lrn_rt in tests:
        data = list(zip(x_data, y_gen(real_p, x_data)))
        start = time.time()
        res = descent(data, cost_func, init_p, lrn_rt=lrn_rt)
        time_elapsed = time.time() - start
        performance.append((res, round(time_elapsed,2)))
    pprint(performance)

run()