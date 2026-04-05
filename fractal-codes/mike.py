import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr

cmap = cmr.chroma
max_iter = 300

c = -0.7 + 0.27j

start = 1j
resX=800    
resY=800

matrix = np.zeros((resX, resY))


def create (z: float, c: float):       
        for i in range(max_iter):
            b = c + -0.5j    
            z = z*z + c + b

            if abs(z) > 2:
                return i
            
        return max_iter


for x in range (resX):
    for y in range (resY):
        z = (x/200 - 2) + (y/200 - 2)*1j
        matrix[y][x] = create(z, c)

plt.imshow(matrix, cmap=cmap)
plt.show()
