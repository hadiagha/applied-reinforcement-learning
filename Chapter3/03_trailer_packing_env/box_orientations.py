import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Define all 6 orientations of a box with base dimensions (l, w, h)
orientations = [
    (3, 2, 1),  # Orientation 0: (l, w, h)
    (3, 1, 2),  # Orientation 1: (l, h, w)
    (2, 3, 1),  # Orientation 2: (w, l, h)
    (2, 1, 3),  # Orientation 3: (w, h, l)
    (1, 3, 2),  # Orientation 4: (h, l, w)
    (1, 2, 3),  # Orientation 5: (h, w, l)
]

labels = [
    "0: (l, w, h)",
    "1: (l, h, w)",
    "2: (w, l, h)",
    "3: (w, h, l)",
    "4: (h, l, w)",
    "5: (h, w, l)"
]

def draw_box(ax, origin, dims, color='skyblue'):
    l, w, h = dims
    x, y, z = origin
    # Define corners of the cuboid
    corners = np.array([
        [x, y, z],
        [x + l, y, z],
        [x + l, y + w, z],
        [x, y + w, z],
        [x, y, z + h],
        [x + l, y, z + h],
        [x + l, y + w, z + h],
        [x, y + w, z + h]
    ])

    # Define the 6 faces using the corners
    faces = [
        [corners[i] for i in [0, 1, 2, 3]],  # bottom
        [corners[i] for i in [4, 5, 6, 7]],  # top
        [corners[i] for i in [0, 1, 5, 4]],  # front
        [corners[i] for i in [2, 3, 7, 6]],  # back
        [corners[i] for i in [1, 2, 6, 5]],  # right
        [corners[i] for i in [0, 3, 7, 4]]   # left
    ]

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='black', alpha=0.9))

fig = plt.figure(figsize=(15, 10))

for i, dims in enumerate(orientations):
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    draw_box(ax, (0, 0, 0), dims)
    ax.set_title(f"Orientation {labels[i]}")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, 4)
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')

plt.tight_layout()
plt.show()
