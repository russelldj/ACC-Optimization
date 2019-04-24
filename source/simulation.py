import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse

#Model params
MAX_ACCEL = 10
MAX_DECCEL = -10
LATENCY = 0
NUM_CARS = 7
MIN_SPEED = 0
MAX_SPEED = 10
ROAD_LEN = 50

#Decision variables
FOLLOWING_DIST = 1
JERK = 2

class Car():
    """
    A class which represents a single car controlled by the ACC system
    """
    def __init__(self, location=0, following_dist=FOLLOWING_DIST, jerk=JERK, max_accel=MAX_ACCEL, max_deccel=MAX_DECCEL, min_speed=MIN_SPEED, max_speed=MAX_SPEED):
        self.location = location
        self.following_dist = following_dist
        self.jerk = jerk
        self.max_accel = max_accel
        self.max_deccel = max_deccel
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed = 1 # this should be updated so it's an input

    def accel_of_dist(self, dist):
        accel = (dist - self.following_dist) * self.jerk # compute the sloped section
        accel = np.min([accel, self.max_accel]) # make sure it's not too big, i.e. create the top horizontal part
        accel = np.max([accel, self.max_deccel]) # make sure it's not too small, i.e. create the bottom horizontal part
        return accel

    def plot_accel_of_dists(self, max_dist=FOLLOWING_DIST*10):
        dists = np.linspace(0, max_dist, num=200)
        accels_of_dist = list()
        for dist in dists:
            accels_of_dist.append(self.accel_of_dist(dist))
        plt.plot(dists, accels_of_dist)
        plt.xlabel("distance")
        plt.ylabel("acceleration")
        plt.show()

    def move(self, dist_to_next, timestep):
        """
        move based on the speed * timestep
        Use the acceleration to update the speed
        """
        accel = self.accel_of_dist(dist_to_next)
        self.location = self.location + self.speed * timestep
        self.speed = self.speed + accel * timestep
        self.speed = max(self.speed, self.min_speed) # we aren't going to go backward
        self.speed = min(self.speed, self.max_speed) # we can't go too fast

class Road():
    def __init__(self, op_dist=FOLLOWING_DIST, jerk=JERK, num_cars=NUM_CARS, timestep = 0.1):
        self.num_cars = num_cars
        self.cars = list()
        self.op_dist = op_dist
        self.jerk = jerk
        self.timestep = timestep # simulation_timestep
        for i in range(num_cars):
            self.cars.append(Car(i, self.op_dist, self.jerk)) # initialize the cars on our roadway

    def plot_car_locations(self):
        car_locations = list()
        for car in self.cars:
            car_locations.append(car.location)
        plt.xlim(0, ROAD_LEN)
        plt.scatter(car_locations, np.zeros_like(car_locations)) #scatter plot of locations with y=0
        plt.pause(0.1) # display
        plt.cla() # clear the points

    def move_cars(self):
        """
        move based on the speed
        """
        #spoofed_dist_to_next = FOLLOWING_DIST * 1.5# this should make them accelerate
        dists_to_next = self.get_dist_to_next()
        for i, car in enumerate(self.cars):
            dist_to_next = dists_to_next[i]
            car.move(dist_to_next, self.timestep)

    def get_dist_to_next(self):
        """
        for each car get the distance to the next car on the road
        """
        dist_for_each_car = list()
        for i, car in enumerate(self.cars):
            current_car_loc = car.location
            dists_to_next = [c.location - current_car_loc for c in self.cars] # this is called list comprehension
            min_dist = np.inf
            for dist in dists_to_next:
                if dist > 0: # we only want cars in front of the current one
                    min_dist = min(min_dist, dist) # find the nearest car

            dist_for_each_car.append(min_dist)
        return dist_for_each_car

    def are_all_past(self, road_len=ROAD_LEN):
        for car in self.cars:
            if car.location < road_len:
                return False # one was still one the road

        return True # none of them were still on the road

def black_box(op_dist, jerk, show_locations=True):
    road = Road(op_dist, jerk)
    num_timesteps = 0
    while not road.are_all_past():
        if show_locations:
            road.plot_car_locations()
        road.move_cars()
        num_timesteps += 1

    return num_timesteps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-dist", default=FOLLOWING_DIST, type=float)
    parser.add_argument("--jerk", default=JERK, type=float)
    parser.add_argument("--show-accel-graph", action="store_true", help="show the car's policy on acceleration versus distance to the nearest car")
    parser.add_argument("--show-simulation", action="store_true", help="show the simulation of the cars driving")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.show_accel_graph:
        temp_location = 0
        Car(temp_location, args.op_dist, args.jerk).plot_accel_of_dists()
    #car = Car(args.op_dist, args.jerk)
    #car.plot_accel_of_dists()
    #road = Road(args.op_dist, args.jerk, NUM_CARS) # make sure you can initialize Road
    #while True:
    #    road.plot_car_locations()
    #    road.move_cars()
    num_timesteps = black_box(args.op_dist, args.jerk, args.show_simulation)
    print("It took {} timesteps for all the cars to clear the road with an optimal following distance of {} and a jerk of {}".format(num_timesteps, args.op_dist, args.jerk))
