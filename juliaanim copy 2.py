import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ----------------------------
# load image
# ----------------------------
img = Image.open("images/mike_8_2.png").convert("RGB")
width, height = img.size
base_img = np.array(img) / 255.0

# ----------------------------
# grid
# ----------------------------
resX, resY = 4000, 4000
x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)

# pixel mapping (fixed reference grid)
px = ((xx + 2) / 4 * (width - 1)).astype(int).clip(0, width - 1)
py = ((yy + 2) / 4 * (height - 1)).astype(int).clip(0, height - 1)

max_iter = 400
frames = 300
angles = np.linspace(0, 2*np.pi, frames)

os.makedirs("morph_frames", exist_ok=True)

# ----------------------------
# fractal function
# ----------------------------
def julia_with_channel(channel, sensitivity, c_base):
    z = xx + 1j * yy
    c_map = c_base + channel * sensitivity

    matrix = np.full((resY, resX), max_iter, dtype=float)
    escaped = np.zeros((resY, resX), dtype=bool)

    for i in range(max_iter):
        mask = ~escaped

        b = np.tanh(z[mask] * np.sinh(z[mask]))
        z[mask] = (z[mask]**2 / (b + 0.5j)) + c_map[mask]

        new_escaped = mask & (np.abs(z) > 30000)
        matrix[new_escaped] = i
        escaped |= new_escaped

        if not mask.any():
            break

    return matrix / max_iter

# ----------------------------
# main loop
# ----------------------------
for frame in range(frames):
    print(f"frame {frame+1}/{frames}")

    t = angles[frame]

    # ----------------------------
    # 1. rotate image
    # ----------------------------
    img_rot = Image.fromarray((base_img * 255).astype(np.uint8))
    img_rot = img_rot.rotate(np.degrees(t), resample=Image.BICUBIC, expand=False)
    img_array = np.array(img_rot) / 255.0

    # ----------------------------
    # 2. warp sampling coords (morph)
    # ----------------------------
    warp_strength = 15

    px_warp = px + warp_strength * np.sin(yy * 3 + t)
    py_warp = py + warp_strength * np.cos(xx * 3 + t)

    px_warp = px_warp.astype(int).clip(0, width - 1)
    py_warp = py_warp.astype(int).clip(0, height - 1)

    # ----------------------------
    # 3. sample RGB channels
    # ----------------------------
    R = img_array[py_warp, px_warp, 0]
    G = img_array[py_warp, px_warp, 1]
    B = img_array[py_warp, px_warp, 2]

    # ----------------------------
    # 4. animated fractal params
    # ----------------------------
    c = 0.7 * np.exp(1j * t)

    sr = 1.0 + 0.4 * np.sin(t)
    sg = 1.2 + 0.4 * np.cos(t)
    sb = 2.3 + 0.4 * np.sin(2*t)

    # ----------------------------
    # 5. compute fractal channels
    # ----------------------------
    matrix_r = julia_with_channel(R, sr, c)
    matrix_g = julia_with_channel(G, sg, c)
    matrix_b = julia_with_channel(B, sb, c)

    rgb_output = np.stack([matrix_r, matrix_g, matrix_b], axis=2)

    # ----------------------------
    # 6. save frame
    # ----------------------------
    out = (rgb_output * 255).astype(np.uint8)
    Image.fromarray(out).save(f"morph_frames/frame_{frame:04d}.png")