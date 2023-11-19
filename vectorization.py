#### I wrote this code as part of my learning journey , Deep Learning Specialization from Deeplearning.ai , Thanks to Andrew Ng ########
import numpy as np  # import statements
import time 

####### testing the speeds between using explicit for loops and vectorization ######
a = np.random.rand(1000000)
b = np.random.rand(1000000)

####### using vectorization ############
start = time.time()    ## gets the starting time 
ans = np.dot(a,b)      ## dot product operator
stop = time.time()     ## gets stopping time

print(ans)
print("time taken :" + str((stop - start)*1000) + " ms")  ## prints the time taken using vectorization

####### using using explicit for loop ############
start = time.time()    ## gets the starting time 
for i in range(1000000):
  ans += a[i]*b[i]
stop = time.time()     ## gets stopping time

print(ans)
print("time taken :" + str((stop - start)*1000) + " ms")  ## prints the time taken using explicit for loop

##### the times obtained will show that using c=vectorization is 300 times faster, it takes advantage of Parallelism in CPU or GPU #########

