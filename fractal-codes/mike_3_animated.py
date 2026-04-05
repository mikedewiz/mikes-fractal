import numpy as np
import cmasher as cmr
from PIL import Image
import os

cmap = cmr.chroma
c = -0.7 + 0.27j
resX, resY = 1600, 1600     # lower res for speed; raise for quality 
total_frames = 270
zoom_start = 1.0
zoom_end = 1e10             # 10 billion× zoom
#target_x, target_y = -0.501951, 0.146064
target_x, target_y = -0.502496, 0.145533

os.makedirs("zoom_frames", exist_ok=True)

def render_frame(cx, cy, zoom):
    half = 2.0 / zoom
    x = np.linspace(cx - half, cx + half, resX)
    y = np.linspace(cy - half, cy + half, resY)
    xx, yy = np.meshgrid(x, y)
    z = xx + 1j * yy
    matrix = np.full((resY, resX), float(max_iter))
    escaped = np.zeros((resY, resX), dtype=bool)

    for i in range(max_iter):
        mask = ~escaped
        b = np.abs(z[mask]) + 0.33
        z[mask] = b * z[mask] ** 2 + c
        new_escaped = mask & (np.abs(z) > 2)
        matrix[new_escaped] = i
        escaped |= new_escaped

    normalized = matrix / max_iter
    colored = (cmap(normalized) * 255).astype(np.uint8)
    return Image.fromarray(colored)

for f in range(total_frames):
    zoom = zoom_start * (zoom_end / zoom_start) ** (f / (total_frames - 1))
    max_iter = int(400 + 200 * np.log10(zoom))
    img = render_frame(target_x, target_y, zoom)
    img.save(f"/zoom_frames/frame_{f:04d}.png")
    print(f"Frame {f+1}/{total_frames}  zoom={zoom:.2e}")