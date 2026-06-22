import numpy as np

from env.constants import GRID_SIZE, FLOOR, HAZARD, WALL, GOAL, TILE_SIZE, IMAGE_SIZE

COLORS = {
    FLOOR: np.array([235, 235, 225], dtype=np.uint8),
    WALL: np.array([45, 45, 50], dtype=np.uint8),
    HAZARD: np.array([220, 55, 55], dtype=np.uint8),
    GOAL: np.array([50, 180, 90], dtype=np.uint8),
}

AGENT_COLOR = np.array([45, 105, 230], dtype=np.uint8)

def render_grid(grid, agent_pos):
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            tile = grid[r, c]
            color = COLORS[tile]

            r0 = r * TILE_SIZE
            r1 = r0 + TILE_SIZE
            c0 = c * TILE_SIZE
            c1 = c0 + TILE_SIZE

            image[r0:r1, c0:c1] = color

    ar, ac = agent_pos
    r0 = ar * TILE_SIZE
    r1 = r0 + TILE_SIZE
    c0 = ac * TILE_SIZE
    c1 = c0 + TILE_SIZE

    image[r0:r1, c0:c1] = AGENT_COLOR

    return image