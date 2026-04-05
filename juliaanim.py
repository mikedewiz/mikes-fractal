import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmasher as cmr

cmap = cmr.rainforest
max_iter = 200

resX, resY = 4000, 4000  # lower for animation speed

x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)

frames = 150
r = 0.7
angles = np.linspace(0, 2*np.pi, frames)

fig, ax = plt.subplots(figsize=(6,6))

def compute_frame(c):
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

    return matrix

def animate(frame):
    ax.set_axis_off()
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])

    c = r * np.exp(1j * angles[frame])  # rotating c

    matrix = compute_frame(c)
    img = ax.imshow(matrix, cmap=cmap)

    return [img]

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50)

anim.save("my_fractal.gif", writer="pillow", fps=30)