This is a repository for the optimization of a cruise control system.

# quickstart 
To experiement with the system go into the `source` directory and run `python simulation.py`.

# more control
If you want to adjust some parameters, you can pass them as command line arguments. These can be listed with `python simulation.py --help`. If you want if you want to adjust the jerk and optimal following distance you can do `python simulation.py --jerk <jerk> --op-dist <op-dist>` where <jerk> and <op-dist> are the values you want for that parameter. If you want to see the graph of acceleration versus distance include the `--show-accel-graph` flag. Finally, if you want to see the simulation progress and not just get the result, you can include the `--show-simulation` flag. For example you could do `python simulation.py --jerk 2 --op-dist 4 --show-accel-graph --show-simulation`

# TODO 
I made an example optimization script in `source/optimization_examples.py`. Currently it is broken and returns the following error message: `Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 54.000000
         Iterations: 2
         Function evaluations: 96
         Gradient evaluations: 21
	[0.99997875 1.99999764]`

# Contour plot
![Contour plot](https://github.com/russelldj/ACC-Optimization/blob/master/contour.png)
