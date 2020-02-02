import numpy as np
import matplotlib.pyplot as plt

"""
Machine learning can be viewed as the tsak of finding a function y = f(x) which correctly assigns 
a vector x to a class y. This script show how to construct a function f(x) for a given set of 
points x and classes y. Naturally, most problem in the field of machine learning are not as simple
since they cannot be modeled simply by using a polynomial function. 
"""


def fit_poly(x, y, degree):
    pol = np.polyfit(x, y, degree)
    xx = np.linspace(min(x), max(x))
    yy = np.polyval(pol, xx)
    return xx, yy


def plot_poly(x, y, xx, yy, degree):
    d = 1
    plt.plot(xx, yy, '-', label=f'deg. {degree}')
    plt.plot(x, y, 'ro')
    plt.axis([min(xx) - d, max(xx) + d, min(yy) - d, max(yy) + d])
    plt.legend()


def main():
    x = np.array([2., 3., 4., 5., 6.])
    y = np.array([2., 6., 5., 5., 6.])

    for d in range(1, len(x)):
        xx, yy = fit_poly(x, y, d)
        plot_poly(x, y, xx, yy, d)

    plt.show()


if __name__ == "__main__":
    main()



