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
    b = np.tanh(z[mask] * np.sinh(z[mask]))
    z[mask] = (z[mask]**2 / (b + 0.5j)) + c
    new_escaped = mask & (np.abs(z) > 10000)
    matrix[new_escaped] = i
    escaped |= new_escaped


plt.imshow(matrix, cmap)

normalized = matrix / max_iter
colored = (cmap(normalized) * 255).astype(np.uint8)
img = Image.fromarray(colored)
#img.save('images/mike_6.png')

plt.show()