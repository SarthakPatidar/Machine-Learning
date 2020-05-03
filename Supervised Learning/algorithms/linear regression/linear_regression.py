from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def create_dataset(num, variance, step=2, correlation=False):
    ys = []
    val = 1
    for i in range(num):
        val += random.randrange(-variance, variance)
        ys.append(val)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    
    xs = [i for i in range(num)]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope(xs, ys) :
    m = (mean(xs)*mean(ys) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs**2))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    ys_mean = [mean(ys_orig) for _ in ys_orig]
    squared_error_y_line = squared_error(ys_orig, ys_line)
    squared_error_mean_line = squared_error(ys_orig, ys_mean)

    return 1 - (squared_error_y_line / squared_error_mean_line)


xs, ys =  create_dataset(35, 40, 2, correlation='pos')

m, b = best_fit_slope(xs, ys)
regression_line = [(m*x + b) for x in xs]

x_predict = 8
y_predict = (m*x_predict) + b


#### R Squared Value ####
print("R squared value = ", coefficient_of_determination(ys, regression_line))

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.scatter(x_predict, y_predict, color='r')
plt.show()