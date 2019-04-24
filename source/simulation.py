import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse

#Model params
MAX_ACCEL = 10
MAX_DECCEL = -10
LATENCY = 0
NUM_CARS = 7

#Decision variables
OP_FOLLOWING_DIST = 8
JERK = 2

class Car():
    """
    A class which represents a single car controlled by the ACC system
    """
    def __init__(self, location=0, op_following_dist=OP_FOLLOWING_DIST, jerk=JERK, max_accel=MAX_ACCEL, max_deccel=MAX_DECCEL):
        self.location = location
        self.op_following_dist = op_following_dist
        self.jerk = jerk
        self.max_accel = max_accel
        self.max_deccel = max_deccel

    def accel_of_dist(self, dist):
        accel = (dist - self.op_following_dist) * self.jerk # compute the sloped section
        accel = np.min([accel, self.max_accel]) # make sure it's not too big, i.e. create the top horizontal part
        accel = np.max([accel, self.max_deccel]) # make sure it's not too small, i.e. create the bottom horizontal part
        return accel

    def plot_accel_of_dists(self, max_dist=OP_FOLLOWING_DIST*3):
        dists = np.linspace(0, max_dist, num=200)
        accels_of_dist = []
        for dist in dists:
            accels_of_dist.append(self.accel_of_dist(dist))
        plt.plot(dists, accels_of_dist)
        plt.xlabel("distance")
        plt.ylabel("acceleration")
        plt.show()

class Road():
    def __init__(self, num_cars=NUM_CARS):
        pass


def initialize_cars(num_cars=NUM_CARS):
    locs = np.linspace()

def simulation(op_following_dist=OP_FOLLOWING_DIST, jerk=JERK, max_accell=MAX_ACCEL, max_deccel=MAX_DECCEL, latency=LATENCY):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-dist", default=OP_FOLLOWING_DIST, type=float)
    parser.add_argument("--jerk", default=JERK, type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    car = Car(args.op_dist, args.jerk)
    car.plot_accel_of_dists()
