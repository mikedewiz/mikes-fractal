import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cmasher as cmr
from PIL import Image
import os

cmap = cmr.rainforest
max_iter = 400
c = 0.7 + 0.1j
resX=1600
resY=1600
total_frames = 270
zoom_start = 1.0
zoom_end = 1e10
target_x, target_y = -0.293530, 0.087520

os.makedirs("mike6-frames", exist_ok=True)

x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)
z = xx + 1j * yy

def render_frame(cx, cy, zoom, max_iter):
    half = 2.0 / zoom
    x = np.linspace(cx - half, cx + half, resX)
    y = np.linspace(cy - half, cy + half, resY)
    xx, yy = np.meshgrid(x, y)
    z = xx + 1j * yy
    matrix = np.full((resY, resX), float(max_iter))
    escaped = np.zeros((resY, resX), dtype=bool)
    for i in range(max_iter):
        mask = ~escaped
        b = np.tanh(z[mask] * np.sinh(z[mask]))
        z[mask] = (z[mask]**2 / (b + 0.5j)) + c
        new_escaped = mask & (np.abs(z) > 10000)
        matrix[new_escaped] = i
        escaped |= new_escaped
    normalized = matrix / max_iter
    colored = (cmap(normalized) * 255).astype(np.uint8)
    return Image.fromarray(colored)

for f in range(total_frames):
    zoom = zoom_start * (zoom_end / zoom_start) ** (f / (total_frames - 1))
    max_iter = int(400 + 200 * np.log10(zoom))  # was 200, now 500
    img = render_frame(target_x, target_y, zoom, max_iter)  # pass it in
    img.save(f"/mike6-frames/frame_{f:04d}.png")
    print(f"Frame {f+1}/{total_frames}  zoom={zoom:.2e}  iters={max_iter}")  # log iters too