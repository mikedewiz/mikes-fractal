
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

cmap = cmr.chroma
max_iter = 400
c = -0.7 + 0.27j
resX, resY = 4000, 4000

x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)
z = xx + 1j * yy
matrix = np.full((resY, resX), max_iter, dtype=float)
escaped = np.zeros((resY, resX), dtype=bool)

for i in range(max_iter):
    mask = ~escaped
    b = abs(z[mask]) + 0.33
    z[mask] = b * z[mask]**2 + c
    new_escaped = mask & (np.abs(z) > 2)
    matrix[new_escaped] = i
    escaped |= new_escaped

fig, ax = plt.subplots()
ax.imshow(matrix, cmap=cmap, extent=[-2, 2, 2, -2])

def onclick(event):
    if event.xdata is not None:
        print(f"target_x, target_y = {event.xdata:.6f}, {event.ydata:.6f}")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()