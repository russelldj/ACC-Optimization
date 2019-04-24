This is a repository for the optimization of a cruise control system.

# quickstart 
To experiement with the system go into the source directory and run `python simulation.py`. Alternatively, if you want to adjust the jerk and optimal following distance you can do `python simulation.py --jerk <jerk> --op-dist <op-dist>` where <jerk> and <op-dist> are the values you want for that parameter. If you want to see the graph of acceleration versus distance include the `--show-accel-graph` flag. For example you could do `python simulation.py --jerk 2 --op-dist 4 --show-accel-graph`
