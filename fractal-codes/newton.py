import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cmasher as cmr
from PIL import Image

cmap = cmr.redshift
max_iter = 300
c = 0.7 + 0.6j

resX=4000
resY=4000

x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)
z = xx + 1j * yy

escaped = np.zeros((resY, resX), dtype=bool)

d = 1 + 1j

roots = [
    (2 + 0.27j)**(1/7) * np.exp(2j * np.pi * k / 7)
    for k in range(7)
]

for i in range (max_iter):

    z = z - (z**7 - (2 + 0.27j)) / (  7*z**6 + 1e-9)

matrix = np.zeros((resY, resX))

for index, root in enumerate(roots):
    # If the distance between the pixel and the root is tiny, it belongs to that root
    is_close = np.abs(z - root) < 0.1
    matrix[is_close] = index

plt.imshow(matrix, cmap)

normalized = matrix / matrix.max()
colored = (cmap(normalized) * 255).astype(np.uint8)
img = Image.fromarray(colored)
img.save('images/newton.png')

plt.show()