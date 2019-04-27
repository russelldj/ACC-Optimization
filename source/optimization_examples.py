import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from simulation import black_box

"example taken from https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html"
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def my_sum(x,y):
    return x + y

def black_box_averager(x, y, num_iters=5):
    value = 0
    for i in range(num_iters):
        value += black_box(x,y)
    return value / float(num_iters)

def black_box_vector_input(following_jerk):
    following, jerk = following_jerk
    return black_box(following, jerk)

def plot(function_, limits=[0,1,0,1], samples=20):
    X_MIN = 0
    X_MAX = 1
    Y_MIN = 2
    Y_MAX = 3
    z = np.zeros((samples,samples))
    x_list = np.linspace(limits[X_MIN], limits[X_MAX], samples)
    y_list = np.linspace(limits[Y_MIN], limits[Y_MAX], samples)
    for j in tqdm(range(samples)): #progess bar
        for i in range(samples):
            z[i,j] = function_(x_list[j],y_list[i])
    plt.contour(x_list, y_list, z)
    plt.xlabel("following distance")
    plt.ylabel("jerk")
    plt.title("contour plot of the time taken, averaged over a few iterations")
    plt.show()

if __name__ == "__main__":
    x0 = np.array([1.0,2.0])
    res = minimize(black_box_vector_input, x0, method='BFGS', options={'disp': True})
    print(res.x)
#plot(black_box_averager, [0,3,0,4],40)

