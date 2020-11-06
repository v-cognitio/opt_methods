import math
import csv
import os


def f1(x):
    return -5 * (x ** 5) + 4 * (x ** 4) - 12 * (x ** 3) + 11 * (x ** 2) - 2 * x + 1


def f2(x):
    return math.log10(x - 2) ** 2 + math.log10(10 - x) ** 2 - math.pow(x, 0.2)


def f3(x):
    return -3 * x * math.sin(0.75 * x) + math.exp(-2 * x)


def f4(x):
    return math.exp(3 * x) + 5 * math.exp(-2 * x)


def f5(x):
    return 0.2 * x * math.log10(x) + (x - 2.3) ** 2


def open_file(method, f):
    filename = method + '/' + f.__str__().split()[1] + '.csv'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    file = open(filename, 'w')
    writer = csv.writer(file)
    writer.writerow(['A', 'B', 'len', 'len / len - 1', 'x1', 'x2', 'f(x1)', 'f(x2)'])
    return writer


def dichotomy(f, a, b, e):
    writer = open_file('dichotomy', f)
    prev_len = b - a

    while b - a > e:
        x1 = (a + b - e / 2) / 2
        x2 = (a + b + e / 2) / 2
        fx1 = f(x1)
        fx2 = f(x2)

        writer.writerow([round(a, 10) for a in [a, b, b - a, (b - a) / prev_len, x1, x2, fx1, fx2]])
        prev_len = b - a

        if fx1 < fx2:
            b = x2
        elif fx1 > fx2:
            a = x1
        else:
            a = x1
            b = x2

    if f(a) < f(b):
        r = (a, f(a))
    else:
        r = (b, f(b))

    return r


def golden_ratio(f, a, b, e):
    writer = open_file('golden', f)
    prev_len = b - a

    c = (math.sqrt(5) - 1) / 2
    x1 = b - c * (b - a)
    x2 = a + c * (b - a)
    fx1 = f(x1)
    fx2 = f(x2)

    while b - a >= e:
        writer.writerow([round(e, 10) for e in [a, b, b - a, (b - a) / prev_len, x1, x2, fx1, fx2]])
        prev_len = b - a

        if fx1 < fx2:
            b = x2
            x2 = x1
            x1 = b - c * (b - a)
            fx2 = fx1
            fx1 = f(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + c * (b - a)
            fx1 = fx2
            fx2 = f(x2)

    if f(a) < f(b):
        r = (a, f(a))
        return r
    else:
        r = (b, f(b))
        return r


def calc_fibb_till(e):
    res = [1, 1]
    while True:
        curr = res[len(res) - 1] + res[len(res) - 2]
        res.append(curr)
        if curr > e:
            return res


def fibb(f, a, b, e):
    writer = open_file('fibb', f)
    prev_len = b - a

    fibs = calc_fibb_till((b - a) / e)
    n = len(fibs) - 1
    k = 0
    x1 = a + fibs[n - 2] / fibs[n] * (b - a)
    x2 = a + fibs[n - 1] / fibs[n] * (b - a)
    fx1 = f(x1)
    fx2 = f(x2)
    while b - a >= e:
        writer.writerow([round(e, 10) for e in [a, b, b - a, (b - a) / prev_len, x1, x2, fx1, fx2]])
        prev_len = b - a

        if fx1 < fx2:
            b = x2
            x2 = x1
            x1 = a + fibs[n - k - 2] / fibs[n - k] * (b - a)
            fx2 = fx1
            fx1 = f(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + fibs[n - k - 1] / fibs[n - k] * (b - a)
            fx1 = fx2
            fx2 = f(x2)
        k = k + 1

    if f(a) < f(b):
        r = (a, f(a))
        return r
    else:
        r = (b, f(b))
        return r


def parabolic(f, a, b, e):
    writer = open_file('parabolic', f)
    prev_len = b - a

    x1 = a
    x3 = b
    x2 = (x1 + x3) / 2

    fx1 = f(x1)
    fx2 = f(x2)
    fx3 = f(x3)

    u = x2 - ((x2 - x1) ** 2 * (fx2 - fx3) - (x2 - x3) ** 2 * (fx2 - fx1)) / \
        (2 * ((x2 - x1) * (fx2 - fx3) - (x2 - x3) * (fx2 - fx1)))
    u1 = x1

    fu = f(u)

    while abs(u - u1) >= e:
        a = x1
        b = x3
        if x2 > u:
            if fu > fx2:
                x1 = u
                fx1 = fu
            else:
                x3 = x2
                x2 = u
                fx2 = fu
                fx3 = fx2
        else:
            if fu > fx2:
                x3 = u
                fx3 = fu
            else:
                x1 = x2
                x2 = u
                fx2 = fu
                fx1 = fx2

        writer.writerow([round(e, 10) for e in [a, b, b - a, (b - a) / prev_len, x1, x2, fx1, fx2]])
        prev_len = b - a

        u1 = u
        u = x2 - ((x2 - x1) ** 2 * (fx2 - fx3) - (x2 - x3) ** 2 * (fx2 - fx1)) / \
            (2 * ((x2 - x1) * (fx2 - fx3) - (x2 - x3) * (fx2 - fx1)))

        fu = f(u)

    r = (u, fu)
    return r


# TODO: do it in cycle with arrays of functions, intervals and methods
print("info: (x, f(x))\n")

print("dich for f1:     ", dichotomy(f1, -0.5, 0.5, 0.00001))
print("golden for f1:   ", golden_ratio(f1, -0.5, 0.5, 0.00001))
print("fibb for f1:     ", fibb(f1, -0.5, 0.5, 0.00001))
print("parabolic for f1:", parabolic(f1, -0.5, 0.5, 0.00001))
print()

print("dich for f2:     ", dichotomy(f2, 6, 9.9, 0.00001))
print("golden for f2:   ", golden_ratio(f2, 6, 9.9, 0.00001))
print("fibb for f2:     ", fibb(f2, 6, 9.9, 0.00001))
print("parabolic for f2:", parabolic(f2, 6, 9.9, 0.00001))
print()

print("dich for f3:     ", dichotomy(f3, 0, 2 * math.pi, 0.00001))
print("golden for f3:   ", golden_ratio(f3, 0, 2 * math.pi, 0.00001))
print("fibb for f3:     ", fibb(f3, 0, 2 * math.pi, 0.00001))
print("parabolic for f3:", parabolic(f3, 0, 2 * math.pi, 0.00001))
print()

print("dich for f4:     ", dichotomy(f4, 0, 1, 0.00001))
print("golden for f4:   ", golden_ratio(f4, 0, 1, 0.00001))
print("fibb for f4:     ", fibb(f4, 0, 1, 0.00001))
print("parabolic for f4:", parabolic(f4, 0, 1, 0.00001))
print()

print("dich for f5:     ", dichotomy(f5, 0.5, 2.5, 0.00001))
print("golden for f5:   ", golden_ratio(f5, 0.5, 2.5, 0.00001))
print("fibb for f5:     ", fibb(f5, 0.5, 2.5, 0.00001))
print("parabolic for f5:", parabolic(f5, 0.5, 2.5, 0.00001))
print()

