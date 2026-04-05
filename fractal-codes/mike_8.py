import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

max_iter = 300
c = -0.2 + 50j
resX, resY = 4000, 4000

x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)
z = xx + 1j * yy

rgb_matrix = np.zeros((resY, resX, 3), dtype=np.uint8)
escaped = np.zeros((resY, resX), dtype=bool)

for i in range(max_iter):
    mask = ~escaped
    if not np.any(mask): break

    b = np.tanh(z[mask]) + np.cosh(z[mask])

    z[mask] = (z[mask]) **5 * (b + c)

    new_escaped = mask & (np.abs(z) > 1000)
    
    if np.any(new_escaped):

        r_vals = (np.abs(z[new_escaped]) * 70) % 256
        g_vals = (np.abs(z[new_escaped]) * 40) % 256
        b_vals = (np.abs(z[new_escaped]) * 80) % 256
        

        rgb_matrix[new_escaped] = np.stack([r_vals, g_vals, b_vals], axis=-1).astype(np.uint8)
    
    escaped |= new_escaped

plt.imshow(rgb_matrix)
img = Image.fromarray(rgb_matrix)
img.save('mike_8_2.png')
plt.show()