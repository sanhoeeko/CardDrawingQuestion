from math import exp

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import comb


def getTransMatrix(n: int):
    diag = np.array(range(1, n + 1)) / n
    down = (1 - diag)[:-1]
    return np.diag(diag) + np.diag(down, -1)


def getBinaryDist(n: int, m: int):
    v = np.zeros((n, 1))
    for i in range(m + 1):
        v[i] = comb(m, i)
    return v / 2 ** n


class CardTest:
    def __init__(self, N, T, start_state=None):
        self.N = N
        self.T = T
        self.M = getTransMatrix(N)
        self.Mpower = self.M.copy()
        if start_state is not None:
            self.state = start_state
        else:
            self.state = np.zeros((self.N, 1))
            self.state[0, 0] = 1
        self.history = [self.state]

    def step(self):
        self.state = self.M @ self.state
        self.history.append(self.state)

    def pack(self):
        self.history = np.hstack(self.history)

    def go(self):
        for t in range(self.T):
            self.step()
        self.pack()

    def show(self):
        plt.imshow(self.history[::-1, :])
        plt.show()

    def side_show(self):
        icolor = cm.BuGn(np.arange(self.history.shape[1]) / self.history.shape[1])
        for i in range(self.T):
            plt.plot(self.history[:, i], color=icolor[i])
        plt.show()

    def ave_var(self):
        aves = []
        vars = []
        x = np.array(range(1, self.N + 1)) / self.N
        x2 = x * x
        lst = self.history if type(self.history) == list else self.history.T
        for v in lst:
            ave = np.dot(x, v)
            moment2 = np.dot(x2, v)
            var = moment2 - ave ** 2
            aves.append(ave)
            vars.append(var)
        setattr(self, 'ave', aves)
        setattr(self, 'var', vars)

    def get_var(self):
        xs = np.array(range(0, self.T + 1)) / self.N
        return xs, self.var

    def matrix_step(self):
        self.Mpower = self.M @ self.Mpower

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


c = CardTest(200, 2000, )
c.go()
c.ave_var()
xs, ys = c.get_var()
print(max(ys))
ys /= max(ys)
opt, cov = curve_fit(vec_theoretical_var, xs, ys, [1, 1, 1.75])
y_fit = vec_theoretical_var(xs, *opt)
print(opt, cov)
plt.plot(xs, ys, xs, y_fit)
plt.legend(['z(t)', 'zeta(t)'])
plt.show()

'''
c = CardTest(100, 1000, )
plt.ion()
for t in range(c.T):
    c.matrix_step()
    plt.cla()
    plt.imshow(c.Mpower)
    plt.pause(0.01)
    plt.show()
'''
