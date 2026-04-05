# Run this once as a separate script to find your target
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

cmap = cmr.rainforest
max_iter = 400
c = 0.7 + 0.1j
resX, resY = 4000, 4000  # low res just for picking

x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)
z = xx + 1j * yy
matrix = np.full((resY, resX), max_iter, dtype=float)
escaped = np.zeros((resY, resX), dtype=bool)

for i in range(max_iter):
    mask = ~escaped
    b = np.tanh(z[mask] * np.sinh(z[mask]))
    z[mask] = (z[mask]**2 / (b + 0.5j)) + c
    new_escaped = mask & (np.abs(z) > 10000)
    matrix[new_escaped] = i
    escaped |= new_escaped

fig, ax = plt.subplots()
ax.imshow(matrix, cmap=cmap, extent=[-2, 2, 2, -2])  # <-- this maps pixels to complex coords

def onclick(event):
    if event.xdata is not None:
        print(f"target_x, target_y = {event.xdata:.6f}, {event.ydata:.6f}")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()