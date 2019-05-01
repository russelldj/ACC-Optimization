import simulation
import matplotlib.pyplot as plt
import pdb
JERK=3
Follow=7

## find the ideal following distance and jerk within the acceptable range so that we can minimizae num_timesteps
#Plot different following distance against time vectorby running the black box through an iteration
#follow_vector=[] #list()
#time=[]
#
#for i in range (12):
#    follow_vector.append(i)
#    time.append(simulation.black_box(i,JERK))
#
#plt.scatter(follow_vector,time)
#plt.show()
#plt.clf()
#print(follow_vector)
#print(time)
follow_vector=[] #list()
times=[]
gass=[]
crashes=[]
# do this again with constant following distance
for i in range (12):
    follow_vector.append(i)
    time, gas, crash = simulation.black_box(Follow,i,False,True)
    times.append(time)
    gass.append(gas)
    crashes.append(crash)

plt.scatter(follow_vector,times)
plt.scatter(follow_vector,gass)
plt.scatter(follow_vector,crashes)
plt.legend(["times","gas","crashes"])
plt.show()

#how to plot and see plot:
#plt.scatter([1,2,5,6,8,8],[7,9,0,2,2,2])
#plt.show()
