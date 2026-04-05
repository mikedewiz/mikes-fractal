import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr

cmap = cmr.rainforest
max_iter = 1000
c = -0.7 + 0.27j

resX=4000
resY=4000

matrix = np.zeros((resX, resY))


def create (z: float, c: float):       
        for i in range(max_iter):    
            z = z*z + c

            if abs(z) > 2:
                return i
            
        return max_iter


for x in range (resX):
    #real_counter=x/100
    for y in range (resY):
        #imag_counter=y/100
        #z = real_counter + imag_counter*1j
        z = (x/200 - 2) + (y/200 - 2)*1j
        
        matrix[y][x] = create(z, c)
        #create(z=z, c=c)

plt.imshow(matrix, cmap=cmap)
plt.show()
