# import math as mt
# import my_algorithms as ma
# import numpy as np
#
#
# class MinSearchAlgorithm:
#
#     def __init__(self, algorithm_function):
#         self.algorithm = algorithm_function
#
#     def find_min(self, my_function, eps):
#         return self.algorithm(my_function.function, my_function.lower_bound, my_function.upper_bound, eps)
#
#
# class SearchResult:
#
#     def __init__(self, point, value, its, a_bounds, b_bounds, x1_points, f1_values, prev_len):
#         self.point = point
#         self.value = value
#         self.its = its
#         self.a_bounds = a_bounds
#         self.b_bounds = b_bounds
#         self.x1_points = x1_points
#         self.f1_values = f1_values
#         self.prev_len = prev_len
#
#
# def func_deriv(func, x):
#     x_inc = 0.0000000001
#     y_inc = func(x + x_inc) - func(x)
#     return y_inc / x_inc
#
#
# def eval_deriv_general(func, x, pos):
#     x_inc = 0.0000000001
#     new = np.array(x)
#     new[pos] += x_inc
#     y_inc = func(*new) - func(*x)
#     return y_inc / x_inc
#
#
# def brent_parabmin(x1, x2, y1, y2):
#     a2 = (y1 - y2) / (x1 - x2)
#     b = (x1 * y2 - x2 * y1) / (x1 - x2)
#     return -b / a2
#
#
# def brent(func, a, c, eps, max_its=50):
#     its = -1
#
#     x = w = v = (a + c) / 2.0
#     fx = fw = fv = func(x)
#     fxd = fwd = fvd = func_deriv(func, x)
#
#     a_bounds, b_bounds, x1_points, f1_values = [a], [c], [], []
#     prev, prev_len = c - a, [c - a]
#     d = e = c - a
#     u1 = u2 = dist_u1 = dist_u2 = 0
#
#     for i in range(max_its):
#         g = e
#         e = d
#         took_u1 = took_u2 = False
#         if x != w and fxd != fwd:
#             u1 = brent_parabmin(x, w, fxd, fwd)
#             took_u1 = a + eps <= u1 <= c - eps and mt.fabs(u1 - x) < g / 2
#             dist_u1 = mt.fabs(u1 - x)
#         if x != v and fxd != fvd:
#             u2 = brent_parabmin(x, v, fxd, fvd)
#             took_u2 = a + eps <= u2 <= c - eps and mt.fabs(u2 - x) < g / 2
#             dist_u2 = mt.fabs(u2 - x)
#         if took_u1 or took_u2:
#             u, dist = (u1, dist_u1) if dist_u1 < dist_u2 else (u2, dist_u2)
#         else:
#             u = (a + x) / 2 if fxd > 0 else (x + c) / 2
#         if mt.fabs(u - x) < eps:
#             u = -eps if u - x < 0 else eps
#             u += x
#
#         d = mt.fabs(x - u)
#         fu, fud = func(u), func_deriv(func, u)
#         x1_points.append(u)
#         f1_values.append(fu)
#
#         if fu <= fx:
#             if u >= x:
#                 a = x
#             else:
#                 c = x
#             v, w, x, fv, fw = w, x, u, fw, fx
#             fx, fvd, fwd, fxd = fu, fwd, fxd, fud
#         else:
#             if u >= x:
#                 c = u
#             else:
#                 a = u
#             if fu <= fw or w == x:
#                 v, w, fv, fw, fvd, fwd = w, u, fw, fu, fwd, fud
#             elif fu <= fv or v == x or v == w:
#                 v, fv, fvd = u, fu, fud
#
#         a_bounds.append(a)
#         b_bounds.append(c)
#         prev_len.append(prev)
#         prev = c - a
#
#         if mt.fabs(x - w) < eps or mt.fabs(func(x) - func(w)) < eps:
#             its = i + 1
#             break
#
#     result = x
#     return result
#
#
# def fact_grad_method_get(func, args, dim, vects, delta, step_search_eps):
#     vals = np.repeat(func(*args), dim)
#     new_args = np.tile(args, (dim, 1)) + vects
#     new_val = np.apply_along_axis(lambda v: func(*v), 1, new_args)
#     grad = (new_val - vals) / delta
#
#     ray = lambda a: func(*(args - a * grad))
#     # step = ma.find_minimum2(ray)
#     step = brent(ray, 0.0, 0.1, step_search_eps)
#     args -= grad * step
#     return args
#
#
# def fact_grad_method(func, start, eps, max_its=500000, print_process=True, step_search_eps=0.00000000001, delta=0.00000000001):
#     dim, args = len(start), np.array(start)
#     val = func(*args)
#     first_steps = {1, 5, 10, 20, 50, 100, 250, 500, 1000, 2000}
#     print_step, its = 5000, max_its
#     ident_array = np.identity(dim)
#     vects = ident_array * delta
#     rel_points = [args]
#
#     for i in range(max_its):
#         if (i in first_steps or i % print_step == 0) and print_process:
#             print('current iteration:', i, 'point:', args, 'val:', func(*args), end='\n', sep='\t\t')
#         args = fact_grad_method_get(func, args, dim, vects, delta, step_search_eps)
#         rel_points.append(np.array(args))
#         new_val = func(*args)
#         if mt.fabs(val - new_val) < eps and i > 50:
#             return args, new_val, i + 1, rel_points
#         val = new_val
#     return args, val, its, rel_points
#
#
# def projected_grad_descent(func, start, eps, point_set, max_its=5000000, print_process=True, step_search_eps=0.0000000001, delta=0.00000000001):
#     dim, args = len(start), np.array(start)
#     val = func(*args)
#     first_steps = {1, 5, 10, 20, 50, 100, 250, 500, 1000, 2000}
#     print_step, its = 5000, max_its
#     ident_array = np.identity(dim)
#     vects = ident_array * delta
#     rel_points = [args]
#
#     for i in range(max_its):
#         if (i in first_steps or i % print_step == 0) and print_process:
#             print('current iteration:', i, 'point:', args, 'val:', func(*args), end='\n', sep='\t\t')
#         args = fact_grad_method_get(func, args, dim, vects, delta, step_search_eps)
#         args = point_set.project(args)
#         rel_points.append(np.array(args))
#         new_val = func(*args)
#         if mt.fabs(val - new_val) < eps and i > 100:
#             return args, new_val, i + 1, rel_points
#         val = new_val
#     return args, val, its, rel_points
#
#
# def fact_grad_method_ravine_args_eval_1(func, args1, args2, rav_step):
#     f1, f2 = func(*args1), func(*args2)
#     args = args1 - rav_step * (args2 - args1) * (f2 - f1) / (np.linalg.norm(args2 - args1) ** 2)
#     return args
#
#
# def fact_grad_method_ravine_args_eval_2(func, x1, x0, rav_step):
#     f1, f0 = func(*x1), func(*x0)
#     args = x1 - rav_step * (x1 - x0) * mt.copysign(1.0, f1 - f0) / np.linalg.norm(x1 - x0)
#     return args
#
#
# def fact_grad_method_ravine_get(func, args, dim, vects, delta, step_search_eps, ngbr_ratio, rav_step, method=2):
#     eval_args = fact_grad_method_ravine_args_eval_1 if method == 1 else fact_grad_method_ravine_args_eval_2
#
#     ngbr_delta = np.random.rand(dim) * ngbr_ratio
#     ngbr = args + ngbr_delta
#     arg_list = np.array([args, ngbr])
#     new_arg_list = np.apply_along_axis(
#         lambda ar: fact_grad_method_get(func, ar, dim, vects, delta, step_search_eps), 1, arg_list)
#     args1, args2 = new_arg_list
#     args = eval_args(func, args1, args2, rav_step)
#     return args
#
#
#
#
# def fact_grad_method_ravine(func, start, eps, max_its=500000, print_process=True, rav_step=0.0001, step_search_eps=0.0000000001, delta=0.00000001):
#     dim, args = len(start), np.array(start)
#     val = func(*args)
#     first_steps = {1, 5, 10, 20, 50, 100, 250, 500, 1000, 2000}
#     print_step, its = 5000, max_its
#     ident_array = np.identity(dim)
#     vects = ident_array * delta
#     ngbr_ratio = 0.00001
#     rel_points = [args]
#
#     for i in range(max_its):
#         if (i in first_steps or i % print_step == 0) and print_process:
#             print('current iteration:', i, 'point:', args, 'val:', func(*args), end='\n', sep='\t\t')
#         args = fact_grad_method_ravine_get(func, args, dim, vects, delta, step_search_eps, ngbr_ratio, rav_step)
#         rel_points.append(np.array(args))
#         new_val = func(*args)
#         if mt.fabs(val - new_val) < eps:
#             return args, new_val, i + 1, rel_points
#         val = new_val
#     return args, val, its, rel_points
#
#
# def fact_coord_descent(func, start, eps, max_its=50000000, print_process=True, delta=0.0000000001):
#     dim, args = len(start), np.array(start)
#     first_steps = {1, 5, 10, 20, 50, 100, 250, 500, 1000, 2000}
#     val = func(*args)
#     print_step, its = 5000, max_its
#     def_step = 1.0
#     rel_points = [args]
#     step, step_decrease, step_growth = def_step, 1.3, 1.3
#     succs_count = 0
#
#     for i in range(max_its):
#         again = False
#         if (i in first_steps or i % print_step == 0) and print_process:
#             print('current iteration:', i, 'point:', args, 'val:', func(*args), end='\n', sep='\t\t')
#
#         val = func(*args)
#         prev_val = val
#         if i > 0 and i % 100 == 0:
#             step = def_step
#         for vect in np.identity(dim):
#             new_arg = np.array(args) + vect * delta
#             new_val = func(*new_arg)
#             deriv = (new_val - val) / delta
#             new_arg -= step * deriv * vect
#             new_val = func(*new_arg)
#             if new_val > val:
#                 succs_count = 0
#                 step /= step_decrease
#                 again = True
#                 break
#             else:
#                 args = new_arg
#                 val = new_val
#                 succs_count += 1
#                 if succs_count == 10:
#                     succs_count = 0
#                     step *= step_growth
#         if again:
#             continue
#         diff = mt.fabs(val - prev_val)
#         rel_points.append(np.array(args))
#         if mt.fabs(diff) * 1000 < eps and i > 200:
#             its = i + 1
#             break
#     return args, val, its, rel_points

import operator
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from prettytable import PrettyTable


def calc_func(func_index, x):
    if func_index == 1:
        return -5 * x ** 5 + 4 * x ** 4 - 12 * x ** 3 + 11 * x ** 2 - 2 * x + 1
    elif func_index == 2:
        return -3 * x * np.sin(0.75 * x) + np.exp(-2 * x)
    elif func_index == 3:
        return np.exp(3 * x) + 5 * np.exp(-2 * x)


def calc_der(func_index, x):
    if func_index == 1:
        return -25 * x ** 4 + 16 * x ** 3 - 36 * x ** 2 + 22 * x - 2
    elif func_index == 2:
        return -2 * np.exp(-2 * x) - 3 * np.sin(0.75 * x) - 2.25 * x * np.cos(0.75 * x)
    elif func_index == 3:
        return np.exp(-2 * x) * (3 * np.exp(5 * x) - 10)


def get_u(x, w, f_x, f_w):
    return (x * f_w - w * f_x) / (f_w - f_x)


def brent(a, c, eps, func_index):
    x = w = v = (a + c) / 2
    fx = fw = fv = calc_func(func_index, x)
    fxd = fwd = fvd = calc_der(func_index, x)
    d = e = c - a
    counter = 0
    prev_u = 0
    while True:
        counter += 1
        g, e = e, d
        u = None
        if x != w and fxd != fwd:
            u = get_u(x, w, fxd, fwd)
            if a + eps <= u <= c - eps and abs(u - x) < g / 2:
                u = u
            else:
                u = None
        if x != v and fxd != fvd:
            u2 = get_u(x, v, fxd, fvd)
            if a + eps <= u2 <= c - eps and abs(u2 - x) < g / 2:
                if u is not None and abs(u2 - x) < abs(u - x):
                    u = u2
        if u is None:
            if fxd > 0:
                u = (a + x) / 2
            else:
                u = (x + c) / 2
        if abs(u - x) < eps:
            u = x + np.sign(u - x) * eps
        d = abs(x - u)
        fu = calc_func(func_index, u)
        fud = calc_der(func_index, u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                c = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
            fvd, fwd, fxd = fwd, fxd, fud
        else:
            if u >= x:
                c = u
            else:
                a = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
                fvd, fwd = fwd, fud
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
                fvd = fud
        print(prev_u - u)
        if counter > 1:
            if abs(prev_u - u) < eps:
                break
        prev_u = u
    return (a + c) / 2, counter

numbers = [1, 2, 3]
intervals = [[-0.5, 0.5], [0, np.pi * 2], [0, 1]]

table = PrettyTable()
table.title = 'Brent'
table.field_names = ['function index', 'Min X', 'Min Y']

for number, interval in zip(numbers, intervals):
    x, counter = brent(interval[0], interval[1], 0.0001, number)
    table.add_row([number, x, '%.10f' % calc_func(number, x)])
print(table.get_string())

for i in numbers:
    epsilons = np.linspace(1, 9, 10)
    iterations = [brent(-0.5, 0.5, 10 ** -eps, i)[1] for eps in epsilons]
    plt.plot(np.log10(10 ** -epsilons), iterations)
    plt.grid()
    plt.ylabel('Количество итераций')
    plt.xlabel('lg(eps)')
    plt.title('Брендт с производной на функции ' + str(i))
    plt.show()


def f1(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def f1_der(x, index):
    return {
        1: 2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1),
        2: 200 * (x[1] - x[0] ** 2)
    }.get(index)


def f2(x):
    return (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def f2_der(x, index):
    return {
        1: 2 * (2 * x[0] ** 3 - 2 * x[0] * x[1] + x[0] - 1),
        2: 2 * (x[1] - x[0] ** 2)
    }.get(index)


def f3(x):
    return (1.5 - x[0] * (1 - x[1])) ** 2 + (2.25 - x[0] * (1 - x[1] ** 2)) ** 2 + (2.625 - x[0] * (1 - x[1] ** 3)) ** 2


def f3_der(x, index):
    return {
        1: 2 * x[0] * (
                x[1] ** 6 + x[1] ** 4 - 2 * x[1] ** 3 - x[1] ** 2 - 2 * x[1] + 3) + 5.25 * x[1] ** 3 + 4.5 * x[
               1] ** 2 + 3 * x[1] - 12.75,
        2: x[0] * (x[0] * (6 * x[1] ** 5 + 4 * x[1] ** 3 - 6 * x[1] ** 2 - 2 * x[1] - 2) + 15.75 * x[1] ** 2 + 9 * x[
            1] + 3)
    }.get(index)


def f4(x):
    return (x[0] + x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4


def f4_der(x, index):
    return {
        1: 2 * (20 * (x[0] - x[3]) ** 3 + x[0] + x[1]),
        2: 2 * (x[0] + 2 * (x[1] - 2 * x[2]) ** 3 + x[1]),
        3: 10 * (x[2] - x[3]) - 8 * (x[1] - 2 * x[2]) ** 3,
        4: 10 * (-4 * (x[0] - x[3]) ** 3 + x[3] - x[2])
    }.get(index)


def get_func(index):
    return {1: f1,
            2: f2,
            3: f3,
            4: f4,
            }.get(index)


def get_func_der(index):
    return {1: f1_der,
            2: f2_der,
            3: f3_der,
            4: f4_der,
            }.get(index)


def get_func_dimension(index):
    return {1: 2,
            2: 2,
            3: 2,
            4: 4,
            }.get(index)


def get_bounds(index):
    return {1: (-1.0, 1.0),
            2: (-1.0, 0.75),
            3: (-3.0, 3.0),
            4: (-1.0, 1.0),
            }.get(index)


def init_approx(index):
    return {1: [-0.5, -0.5],
            2: [-0.5, -0.5],
            3: [-0.5, -0.5],
            4: [-0.5, -0.5, -0.5, -0.5],
            }.get(index)

num_funcs = 3

def coordinate_descent(x, func_number, eps, learning_rate):
    coordinates = []
    n_arg = len(x)
    cycler = cycle([i + 1 for i in range(n_arg)])
    counter = 0
    while True:
        counter += 1
        index = next(cycler)
        d_x = get_func_der(func_number)(x, index)
        new_x = x.copy()
        new_x[index - 1] -= learning_rate * d_x
        coordinates.append([x, new_x])
        if np.linalg.norm(np.array(new_x) - np.array(x)) < eps:
            x = new_x
            break
        x = new_x
    return x, counter, coordinates

table = PrettyTable()
table.title = 'Coordinate descent'
table.field_names = ['function index', 'Min X', 'Min Y']
for i in range(1, num_funcs + 1):
    x, counter, coordinates = coordinate_descent(init_approx(i), i, 0.00001, 0.002)
    table.add_row([i, x, '%.10f' % get_func(i)(x)])
x, counter, coordinates = coordinate_descent(init_approx(4), 4, 0.00001, 0.002)
table.add_row([4, x, '%.10f' % get_func(4)(x)])
print(table.get_string())

for i in range(1, num_funcs + 1):
    delta = 0.025
    bounds = get_bounds(i)
    x = np.arange(bounds[0], bounds[1], delta)
    y = np.arange(bounds[0], bounds[1], delta)
    X, Y = np.meshgrid(x, y)
    Z = get_func(i)([X, Y])
    fig, ax = plt.subplots()
    plt.title('График покоординатного спуска для функции ' + str(i))
    cs = ax.contour(X, Y, Z)
    ax.clabel(cs)
    x, counter, coordinates = coordinate_descent(init_approx(i), i, 0.00001, 0.002)
    lc = LineCollection(coordinates)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.show()

for i in range(1, num_funcs + 2):
    epsilons = np.linspace(1, 10, 10)
    iterations = []
    for eps in epsilons:
        arg = [-1.0, -1.0]
        if i == 4:
            arg = [-1.0, -1.0, -1.0, -1.0]
        x, counter, coordinates = coordinate_descent(arg, i, 10 ** -eps, 0.002)
        iterations.append(counter)
    plt.plot(np.log10(10 ** -epsilons), iterations)
    plt.grid()
    plt.ylabel('Количество итераций')
    plt.xlabel('lg(eps)')
    plt.title('Покоординатный спуск на функции ' + str(i))
    plt.show()


def golden_section(a, b, func, eps):
    K = (1 + np.sqrt(5)) / 2
    x1 = b - (b - a) / K
    x2 = a + (b - a) / K
    f1 = func(x1)
    f2 = func(x2)
    counter = 0
    while abs(b - a) > eps:
        counter += 1
        if f1 < f2:
            b = x2
            f2, x2 = f1, x1
            x1 = b - (b - a) / K
            f1 = func(x1)
        else:
            a = x1
            f1, x1 = f2, x2
            x2 = a + (b - a) / K
            f2 = func(x2)
    return (a + b) / 2


def steepest_descent(x, func_index, eps):
    func = get_func(func_index)
    func_der = get_func_der(func_index)
    variable_number = get_func_dimension(func_index)
    x = [0] * variable_number
    points = [(x, func(x))]
    coordinates = []
    counter = 0
    while True:
        counter += 1
        grad = [func_der(x, i) for i in range(1, variable_number + 1)]
        new_x_func = lambda step: list(map(operator.sub, x, [step * grad[i] for i in range(len(grad))]))
        step_func = lambda step: func(new_x_func(step))
        result_step = golden_section(0, 10, step_func, 10 ** -5)
        new_x = new_x_func(result_step)
        coordinates.append([x, new_x])
        x = new_x
        points.append((x, func(x)))
        if abs(points[-1][1] - points[-2][1]) < eps:
            break
    return coordinates, points, counter


table = PrettyTable()
table.title = 'Steepest descent'
table.field_names = ['function index', 'Min X', 'Min Y']
for i in range(1, num_funcs + 1):
    coordinates, points, counter = steepest_descent(init_approx(i), i, 10 ** -9)
    table.add_row([i, points[-1][0], '%.15f' % points[-1][1]])
coordinates, points, counter = steepest_descent(init_approx(4), 4, 10 ** -9)
table.add_row([4, points[-1][0], '%.15f' % points[-1][1]])
print(table.get_string())

for i in range(1, num_funcs + 1):
    delta = 0.025
    bounds = get_bounds(i)
    x = np.arange(bounds[0], bounds[1], delta)
    y = np.arange(bounds[0], bounds[1], delta)
    X, Y = np.meshgrid(x, y)
    Z = get_func(i)([X, Y])
    fig, ax = plt.subplots()
    plt.title('График наискорейшего спуска для функции ' + str(i))
    cs = ax.contour(X, Y, Z)
    ax.clabel(cs)
    coordinates, points, counter = steepest_descent(init_approx(i), i, 10 ** -9)
    lc = LineCollection(coordinates)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.show()

for i in range(1, num_funcs + 2):
    iterations = []
    epsilons = np.linspace(1, 10, 10)
    for eps in epsilons:
        arg = [0.0, 0.0]
        if i == 4:
            arg = [0.0, 0.0, 0.0, 0.0]
        coordinates, points, counter = steepest_descent(arg, i, 10 ** -eps)
        iterations.append(counter)
    plt.plot(np.log10(10 ** -epsilons), iterations)
    plt.grid()
    plt.ylabel('Количество итераций')
    plt.xlabel('lg(eps)')
    plt.title('Наискорейший спуск на функции ' + str(i))
    plt.show()