import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.cm as cm

from simulation import black_box

accumulator = list()


def f(x):
    # Store the list of function calls
    accumulator.append(x)
    return np.sqrt(x[0]**2 + x[1]**2)
"example taken from https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html"
def rosen(x):
    """The Rosenbrock function"""
    accumulator.append(x)
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen2d(x,y):
    return rosen(np.array([x,y]))


def my_sum(x,y):
    return x + y

def black_box_averager(x, y, num_iters=5):
    value = 0
    for i in range(num_iters):
        value += black_box(x,y)
    return value / float(num_iters)

def black_box_vector_input(following_jerk): #First the optimizer determines the best values but they come out into two scalar variables, in order to put it in the black box to evaluate it, it needs to be in vector form which we put it in here
    accumulator.append(following_jerk)  #saves x[i] at current iteration
    following, jerk = following_jerk
    return black_box(following, jerk)

def plot(function_, limits=[0,1,0,1], samples=20, is_vector=False):
    X_MIN = 0
    X_MAX = 1
    Y_MIN = 2
    Y_MAX = 3
    z = np.zeros((samples,samples))
    x_list = np.linspace(limits[X_MIN], limits[X_MAX], samples)
    y_list = np.linspace(limits[Y_MIN], limits[Y_MAX], samples)
    for j in tqdm(range(samples)): #progess bar
        for i in range(samples):
            if is_vector:
                z[i,j] = function_(np.array([x_list[j], y_list[i]]))
            else:
                z[i,j] = function_(x_list[j],y_list[i])

    colors= plt.contour(x_list, y_list, z)
    plt.colorbar(colors)
    plt.xlabel("following distance")
    plt.ylabel("jerk")
    plt.title("contour plot of the time taken, averaged over a few iterations")
    plt.show()

if __name__ == "__main__":
    x0 = np.array([6,0.5])
    res = minimize(f, x0, method='nelder-mead',
                   options={'xtol': 1e-8, 'disp': True})
    print(res.x)
    accumulator_np = np.array(accumulator)
    colors = cm.rainbow(np.linspace(0, 1, len(accumulator)))
    plt.scatter(accumulator_np[:,0], accumulator_np[:,1],color=colors)

    #plt.legend(loc='upper left')
    #plot(black_box, [min(accumulator_np[:,0]),max(accumulator_np[:,0]),min(accumulator_np[:,1]),max(accumulator_np[:,1])],40)
    plot(f, [-8,8,-8,8],20, True) #replace black_box with rosen2d
