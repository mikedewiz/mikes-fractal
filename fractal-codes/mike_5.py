import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cmasher as cmr
from PIL import Image

cmap = cmr.rainforest
max_iter = 500
c = 0.7 + 0.1j

resX=4000
resY=4000

x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)
z = xx + 1j * yy

matrix = np.full((resY, resX), max_iter, dtype=float)
escaped = np.zeros((resY, resX), dtype=bool)


for i in range (max_iter):
    mask = ~escaped
    b = z[mask]*np.cosh(z[mask])/z[mask]
    z[mask] = z[mask]**4 + c * b
    new_escaped = mask & (np.abs(z) > 2000)
    matrix[new_escaped] = i
    escaped |= new_escaped


plt.imshow(matrix, cmap)

normalized = matrix / max_iter
colored = (cmap(normalized) * 255).astype(np.uint8)
img = Image.fromarray(colored)
img.save('images/mike_5.png')

plt.show()