from dtaidistance import dtw
import numpy as np

#a = [1,2,3,4,5,6,7,8,9]
#b = [2,3,4,5,6,7,8,9,10]

a = [1,1,1,1,1]
b = [2,2,2,2,]
print(a)
print(b)

distance1 = dtw.distance(a,b)

print(distance1)

aa = np.concatenate((a,a))
bb = np.concatenate((b,b))

print(aa)
print(bb)
distance2 = dtw.distance(aa,bb)

print(distance2)