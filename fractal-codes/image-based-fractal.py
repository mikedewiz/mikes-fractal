import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open("images/mike_6.png").convert("RGB").resize((4000,4000))
width, height = img.size
img_array = np.array(img) / 255.0

max_iter = 300
c = 0.7 + 0.1j

x = np.linspace(-2, 2, width)
y = np.linspace(-2, 2, height)
xx, yy = np.meshgrid(x, y)
z = xx + 1j * yy


px = ((xx + 2) / 4 * (width - 1)).astype(int).clip(0, width - 1)
py = ((yy + 2) / 4 * (height - 1)).astype(int).clip(0, height - 1)


R = img_array[py, px, 0]
G = img_array[py, px, 1]
B = img_array[py, px, 2]

sensitivity_r = 1.2
sensitivity_g = 0.8
sensitivity_b = 0.7

def julia_with_channel(channel, sensitivity, c_base, max_iter):
    z = xx + 1j * yy
    c_map = c_base + channel * sensitivity
    matrix = np.full((height, width), max_iter, dtype=float)
    escaped = np.zeros((height, width), dtype=bool)

# c_map[mask]

    for i in range(max_iter):
        mask = ~escaped
        b = np.tanh(z[mask] * np.sinh(z[mask]))
        z[mask] = (z[mask]**2 / (b + 0.5j))+ c_map[mask]
        new_escaped = mask & (np.abs(z) > 30000)
        matrix[new_escaped] = i
        escaped |= new_escaped

    return matrix / max_iter

matrix_r = julia_with_channel(R, sensitivity_r, c, max_iter)
matrix_g = julia_with_channel(G, sensitivity_g, c, max_iter)
matrix_b = julia_with_channel(B, sensitivity_b, c, max_iter)

rgb_output = np.stack([matrix_r, matrix_g, matrix_b], axis=2).astype(np.float32)

plt.imsave("mike_6_colored_test.png", rgb_output)

plt.figure(figsize=(12, 8))
plt.imshow(rgb_output)
plt.axis("off")
plt.tight_layout()
plt.show()