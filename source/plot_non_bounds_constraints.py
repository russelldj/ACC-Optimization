"""
Optimization with constraints
================================

An example showing how to do optimization with general constraints using
SLSQP and cobyla.
"""
import pdb
import numpy as np
import pylab as pl
from scipy import optimize

def f(x):
    # Store the list of function calls
    accumulator.append(x)
    return np.sqrt(x[0]**2 + x[1]**2)

x, y = np.mgrid[-2.03:4.2:.04, -1.6:3.2:.04]
x = x.T
y = y.T

pl.figure(1, figsize=(3, 2.5))
pl.clf()
pl.axes([0, 0, 1, 1])

contours = pl.contour(np.sqrt((x)**2 + (y)**2),
                    extent=[-2.03, 4.2, -1.6, 3.2],
                    cmap=pl.cm.gnuplot)
pl.clabel(contours,
        inline=1,
        fmt='%1.1f',
        fontsize=14)
#pl.plot([-1.5,    0,  1.5,    0, -1.5],
#        [   0,  1.5,    0, -1.5,    0], 'k', linewidth=2)
#pl.fill_between([ -1.5,    0,  1.5],
#                [    0, -1.5,    0],
#                [    0,  1.5,    0],
#                color='.8')
pl.axvline(0, color='k')
pl.axhline(0, color='k')

pl.text(-.9, 2.8, '$x_2$', size=20)
pl.text(3.6, -.6, '$x_1$', size=20)
pl.axis('tight')
pl.axis('off')

# And now plot the optimization path
accumulator = list()


def constraint1(x):
    return np.atleast_1d(x[0] + x[1]) #equal to zero

def constraint2(x):
    return np.atleast_1d(x[1] - 1) #greater than or equal to zero

res = optimize.minimize(f, np.array([7, 3]), method="SLSQP",
                     constraints=[{"fun": constraint1, "type": "eq"},{"fun": constraint2, "type": "ineq"}])
accumulated = np.array(accumulator)
pl.plot(accumulated[:, 0], accumulated[:, 1])
#pl.grid(linestyle='-', linewidth='0.5', color='red') # show the gridlines

print(res.x)
pl.show()
