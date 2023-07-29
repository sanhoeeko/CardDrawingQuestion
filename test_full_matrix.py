from math import exp, log

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit


def getTransMatrix(n: int):
    diag = np.array(range(1, n + 1)) / n
    down = (1 - diag)[:-1]
    return np.diag(diag) + np.diag(down, -1)


class CardTest:
    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.M = getTransMatrix(N)
        self.Mpower = self.M.copy()
        self.history = [self.Mpower]

    def ave_var(self):
        aves = []
        vars = []
        x = np.array(range(1, self.N + 1)) / self.N
        x2 = x * x
        for v in self.history:
            ave = np.dot(x, v)
            moment2 = np.dot(x2, v)
            var = moment2 - ave ** 2
            aves.append(ave)
            vars.append(var)
        setattr(self, 'ave', aves)
        setattr(self, 'var', vars)

    def get_var(self):
        xss = []
        for i in range(self.N - 1):
            x0 = (i + 1) / self.N
            xs = np.array(range(1, self.T + 2)) / self.N - log(1 - x0)
            xss.append(xs)
        var_cache = np.array(self.var)
        self.var = list(np.array(self.var).T)
        return np.array(xss), var_cache

    def matrix_step(self):
        self.Mpower = self.M @ self.Mpower
        self.history.append(self.Mpower)

    def matrix_go(self):
        for t in range(self.T):
            self.matrix_step()


def DebyeF(y: float, a):
    def g(x: float, a):
        if x == 0: return 0
        return x ** a / (exp(x) - 1)

    return quad(lambda x: g(x, a), 0, y)[0]


# curve_fit的函数句柄的第一个入参必须是x
def theoretical_var(x: float, a, b, c):
    return a * DebyeF(b * x, c) * exp(-x)


def vec_theoretical_var(x: np.ndarray, a, b, c):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = theoretical_var(x[i], a, b, c, )
    return y


def fit(xss: list, vars: list):
    n = len(xss) - 1  # 最后一组没法演化，不考虑
    params = []
    errors = []
    for i in range(n):
        try:
            ys = vars[i] / max(vars[i])
            opt, cov = curve_fit(vec_theoretical_var, xss[i], ys, [1, 1, 1])
            errors.append(np.trace(cov))
            params.append(opt)
            print(i)
        except:
            print('fuck!', i)
            break
    return params, errors

def plot_ave(xss: list, aves: list):
    n = len(xss) - 1
    for i in range(n):
        plt.plot(xss[i], aves[i])


c = CardTest(200, 2000, )
c.matrix_go()
c.ave_var()
xs, ys = c.get_var()
# plot_ave(list(xs), list(np.array(c.ave).T/c.N))
# params, errors = fit(list(xs), c.var)
'''
y = c.var[2]
y /= max(y)
x = xs[2]
p = [3.6616,1.5587,1.9854]
y_pred = vec_theoretical_var(x, *p)
plt.plot(x, y, x, y_pred)
plt.show()
'''
