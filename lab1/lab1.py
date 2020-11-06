import math


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


def dichotomy(f, a, b, e):
    while b - a > e:
        x1 = (a + b - e / 2) / 2
        x2 = (a + b + e / 2) / 2
        fx1 = f(x1)
        fx2 = f(x2)
        if fx1 < fx2:
            b = x2
        elif fx1 > fx2:
            a = x1
        else:
            a = x1
            b = x2

    if f(a) < f(b):
        r = (a, f(a))
        return r
    else:
        r = (b, f(b))
        return r


def golden_ratio(f, a, b, e):
    c = (math.sqrt(5) - 1) / 2
    x1 = b - c * (b - a)
    x2 = a + c * (b - a)
    fx1 = f(x1)
    fx2 = f(x2)
    while b - a >= e:
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
    fibs = calc_fibb_till((b - a) / e)
    n = len(fibs) - 1
    k = 0
    x1 = a + fibs[n - 2] / fibs[n] * (b - a)
    x2 = a + fibs[n - 1] / fibs[n] * (b - a)
    fx1 = f(x1)
    fx2 = f(x2)
    while b - a >= e:
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
    x1 = a
    x3 = b
    x2 = (x1 + x3) / 2

    fx1 = f(x1)
    fx2 = f(x2)
    fx3 = f(x3)

    u = x2 - ((x2 - x1) ** 2 * (fx2 - fx3) - (x2 - x3) ** 2 * (fx2 - fx1)) / \
        (2 * ((x2 - x1) * (fx2 - fx3) - (x2 - x3) * (fx2 - fx1)))
    u1 = x1

    while abs(u - u1) >= e:
        if x2 > u:
            if f(u) > f(x2):
                x1 = u
                x2 = (x1 + x3) / 2
            else:
                x3 = x2
                x2 = (x1 + x3) / 2
        else:
            if f(u) > f(x2):
                x3 = u
                x2 = (x1 + x3) / 2
            else:
                x1 = x2
                x2 = (x1 + x3) / 2

        fx1 = f(x1)
        fx2 = f(x2)
        fx3 = f(x3)

        u1 = u
        u = x2 - ((x2 - x1) ** 2 * (fx2 - fx3) - (x2 - x3) ** 2 * (fx2 - fx1)) / \
            (2 * ((x2 - x1) * (fx2 - fx3) - (x2 - x3) * (fx2 - fx1)))

    r = (u, f(u))
    return r


def parabmin(x1, x2, x3, fx1, fx2, fx3):
    u = x2 - ((x2 - x1) ** 2 * (fx2 - fx3) - (x2 - x3) ** 2 * (fx2 - fx1)) / \
        (2 * ((x2 - x1) * (fx2 - fx3) - (x2 - x3) * (fx2 - fx1)))
    return u


def brent(f, a, b, e):
    eps = e
    c = b
    x = (a + c)/2
    w = x
    v = w
    fx = f(x)
    fw = fx
    fv = fw
    k = (3 - math.sqrt(5))/2
    d = c - a
    e = d
    u = parabmin(a, x, b, f(a), fx, f(b))
    while c - a >= eps:
        g = e
        e = d
        un = u
        if x != w and w != v and x != v or fx != fw and fw != fv and fx != fv:
            xu = sorted([v, x, w])
            fxu = sorted([fv, fx, fw])
            x1 = xu[0]
            x2 = xu[1]
            x3 = xu[2]
            fx1 = fxu[0]
            fx2 = fxu[1]
            fx3 = fxu[2]
            un = parabmin(x1, x2, x3, fx1, fx2, fx3)
        if un >= a + eps and un <= c - eps and abs(un-x) < g/2:
            u = un
            d = abs(u - x)
        else:
            if x < (c - a)/2:
                u = x + k * (c - x)
                d = c - x
            else:
                u = x - k * (x - a)
                d = x - a
        if d < eps:
            u = x + math.copysign(1, u - x) * eps
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                c = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u >= x:
                c = x
            else:
                a = x
            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    return u, fu


# TODO: do it in cycle with arrays of functions, intervals and methods
print("info: (x, f(x))\n")

print("dich for f1:     ", dichotomy(f1, -0.5, 0.5, 0.00001))
print("golden for f1:   ", golden_ratio(f1, -0.5, 0.5, 0.00001))
print("fibb for f1:     ", fibb(f1, -0.5, 0.5, 0.00001))
print("parabolic for f1:   ", parabolic(f1, -0.5, 0.5, 0.00001))
print("brent for f1:   ", brent(f1, -0.5, 0.5, 0.00001))
print()

print("dich for f2:     ", dichotomy(f2, 6, 9.9, 0.00001))
print("golden for f2:   ", golden_ratio(f2, 6, 9.9, 0.00001))
print("fibb for f2:     ", fibb(f2, 6, 9.9, 0.00001))
print("parabolic for f2:", parabolic(f2, 6, 9.9, 0.00001))
print("brent for f2:   ", brent(f2, 6, 9.9, 0.00001))
print()

print("dich for f3:     ", dichotomy(f3, 0, 2 * math.pi, 0.00001))
print("golden for f3:   ", golden_ratio(f3, 0, 2 * math.pi, 0.00001))
print("fibb for f3:     ", fibb(f3, 0, 2 * math.pi, 0.00001))
print("parabolic for f3:", parabolic(f3, 0, 2 * math.pi, 0.00001))
print("brent for f3:   ", brent(f3, 0, 2 * math.pi, 0.00001))
print()

print("dich for f4:     ", dichotomy(f4, 0, 1, 0.00001))
print("golden for f4:   ", golden_ratio(f4, 0, 1, 0.00001))
print("fibb for f4:     ", fibb(f4, 0, 1, 0.00001))
print("parabolic for f4:", parabolic(f4, 0, 1, 0.00001))
print("brent for f4:   ", brent(f4, 0, 1, 0.00001))
print()

print("dich for f5:     ", dichotomy(f5, 0.5, 2.5, 0.00001))
print("golden for f5:   ", golden_ratio(f5, 0.5, 2.5, 0.00001))
print("fibb for f5:     ", fibb(f5, 0.5, 2.5, 0.00001))
print("parabolic for f5:", parabolic(f5, 0.5, 2.5, 0.00001))
print("brent for f5:   ", brent(f5, 0.5, 2.5, 0.00001))
print()

