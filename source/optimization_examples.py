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
    #print("running black box vector input with following_jerk = {}".format(following_jerk))
    accumulator.append(following_jerk)  #saves x[i] at current iteration
    following, jerk = following_jerk
    return black_box(following, jerk)

def black_box_vector_input_plus_penalty(following_jerk): #First the optimizer determines the best values but they come out into two scalar variables, in order to put it in the black box to evaluate it, it needs to be in vector form which we put it in here
    #print("running black box vector input with following_jerk = {}".format(following_jerk))
    accumulator.append(following_jerk)  #saves x[i] at current iteration
    following, jerk = following_jerk
    function_cost = black_box(following, jerk)
    penalty_cost = penalty(following_jerk)
    return function_cost + penalty_cost

def plot(function_, limits=[0,1,0,1], samples=20, is_vector=False,load=True):
    X_MIN = 0
    X_MAX = 1
    Y_MIN = 2
    Y_MAX = 3
    z = np.zeros((samples,samples))
    x_list = np.linspace(limits[X_MIN], limits[X_MAX], samples)
    y_list = np.linspace(limits[Y_MIN], limits[Y_MAX], samples)
    if load:
        z = np.load("z_values.npy")
    else:
        for j in tqdm(range(samples)): #progess bar
            for i in range(samples):
                if is_vector:
                    z[i,j] = function_(np.array([x_list[j], y_list[i]]))
                else:
                    z[i,j] = function_(x_list[j],y_list[i])
    np.save("z_values.npy", z) # save these values to avoid recomputing
    colors = plt.contourf(x_list, y_list, z, 30)
    plt.colorbar(colors)
    plt.xlabel("following distance")
    plt.ylabel("jerk")
    plt.title("contour plot of the time taken, averaged over a few iterations")
    #plt.show() Show at the end

def constraint1(x):
    return np.atleast_1d(x[0] - x[1]) #equal to zero

def constraint2(x):
    return np.atleast_1d(x[1] - 1) #greater than or equal to zero

def penalty(x, scale_factor=1000):
    p1 = min(constraint1(x),0)
    p2 = min(constraint2(x),0)
    return (p1 ** 2 + p2 ** 2) * scale_factor # the sum of the sqared penalties

def nelder_mead_unconstrained():
    x0 = np.array([6.0,4.0])
    res = minimize(black_box_vector_input, x0, method='nelder-mead', options={'maxiter': 100})
    return res

def nelder_mead_penalty():
    x0 = np.array([6.0,4.0])
    res = minimize(black_box_vector_input, x0, method='nelder-mead', options={'maxiter': 100})
    return res

def plot_func_plus_penalty(load_saved=True):
    plot(black_box_vector_input_plus_penalty, [0.5,8,0.5,8],10, True, load_saved)

if __name__ == "__main__":
    x0 = np.array([6.0,4.0])
    #res = minimize(f, x0, method='SLSQP', constraints=[{"fun": constraint2, "type": "ineq"}])
    #res = minimize(black_box_vector_input, x0, method='nelder-mead', constraints=[{"fun": constraint1, "type": "ineq"}, {"fun": constraint2, "type": "ineq"}], options={'maxiter': 10})
    plot_func_plus_penalty(False)
    res = nelder_mead_unconstrained()
    print(res.x)
    accumulator_np = np.array(accumulator)
    print(accumulator_np)
    colors = cm.rainbow(np.linspace(0, 1, len(accumulator)))
    plt.scatter(accumulator_np[:,0], accumulator_np[:,1],color=colors)
    plt.show()

    #plt.legend(loc='upper left')
    #plot(black_box, [min(accumulator_np[:,0]),max(accumulator_np[:,0]),min(accumulator_np[:,1]),max(accumulator_np[:,1])],40)
    #plot(black_box_vector_input, [0.5,8,0.5,8],10, True) #replace black_box with rosen2d
