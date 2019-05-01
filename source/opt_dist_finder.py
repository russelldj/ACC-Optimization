import simulation
import matplotlib.pyplot as plt
import pdb
JERK=3
Follow=7

## find the ideal following distance and jerk within the acceptable range so that we can minimizae num_timesteps
#Plot different following distance against time vectorby running the black box through an iteration
follow_vector=[] #list()
time=[]

for i in range (12):
    follow_vector.append(i)
    time.append(simulation.black_box(i,JERK))

plt.scatter(follow_vector,time)
plt.show()
plt.clf()
print(follow_vector)
print(time)
follow_vector=[] #list()
times=[]
gas=[]
# do this again with constant following distance
for i in range (12):
    follow_vector.append(i)
    t, g = simulation.black_box(Follow,i,False,True) # get the time taken and gas used
    times.append(t)
    gas.append(g)

plt.scatter(follow_vector,times)
plt.scatter(follow_vector,gas)
print(gas)
plt.legend(["times", "gas"])
plt.show()

#how to plot and see plot:
#plt.scatter([1,2,5,6,8,8],[7,9,0,2,2,2])
#plt.show()
