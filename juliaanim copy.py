import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from PIL import Image

# --- load image ---
img = Image.open("images/mike_8_2.png").convert("RGB")
width, height = img.size
img_array = np.array(img) / 255.0

# --- grid ---
resX, resY = 400, 400  # lower for animation
x = np.linspace(-2, 2, resX)
y = np.linspace(-2, 2, resY)
xx, yy = np.meshgrid(x, y)

# --- pixel mapping ---
px = ((xx + 2) / 4 * (width - 1)).astype(int).clip(0, width - 1)
py = ((yy + 2) / 4 * (height - 1)).astype(int).clip(0, height - 1)

R = img_array[py, px, 0]
G = img_array[py, px, 1]
B = img_array[py, px, 2]

max_iter = 300

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

# --- animation setup ---
frames = 30
angles = np.linspace(0, 2*np.pi, frames)

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(0,0,1,1)

def animate(frame):
    ax.clear()
    ax.set_axis_off()

    # --- your animation params ---
    r = 0.7
    c = r * np.exp(1j * angles[frame])

    sr = 1.0 + 0.5*np.sin(angles[frame])
    sg = 1.2 + 0.5*np.cos(angles[frame])
    sb = 2.3 + 0.5*np.sin(2*angles[frame])

    matrix_r = julia_with_channel(R, sr, c)
    matrix_g = julia_with_channel(G, sg, c)
    matrix_b = julia_with_channel(B, sb, c)

    rgb_output = np.stack([matrix_r, matrix_g, matrix_b], axis=2)

    img = (rgb_output * 255).astype(np.uint8)
    Image.fromarray(img).save(f"mike6-colorframes/frame_{frame:04d}.png")

    # optional preview
    im = ax.imshow(rgb_output)
    return [im]

#anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50)

#anim.save("rgb_fractal_new.gif", writer="pillow", fps=30)

for frame in range(frames):
    print(f"frame {frame+1}/{frames}")
    animate(frame)