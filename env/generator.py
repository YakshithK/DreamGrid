import numpy as np

from env.constants import GRID_SIZE, FLOOR, HAZARD, WALL, GOAL
from env.pathfinding import shortest_path

def generate_map(seed=None, wall_prob=0.18, num_hazards=5):
    rng = np.random.default_rng(seed)

    while True:
        grid = np.full((GRID_SIZE, GRID_SIZE), FLOOR, dtype=np.int64)

        start = (
            int(rng.integers(0, GRID_SIZE)),
            int(rng.integers(0, GRID_SIZE))
        )

        goal = (
            int(rng.integers(0, GRID_SIZE)), 
            int(rng.integers(0, GRID_SIZE))
        )

        if goal == start:
            continue

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                pos = (r, c)
                if pos == start or pos == goal:
                    continue

                if rng.random() < wall_prob:
                    grid[r, c] = WALL

        placed = 0

        while placed < num_hazards:
            pos = (
                int(rng.integers(0, GRID_SIZE)),
                int(rng.integers(0, GRID_SIZE))
            )
            r, c = pos

            if  pos == start or pos == goal:
                continue
            if grid[r, c] != FLOOR:
                continue

            grid[r, c] = HAZARD
            placed += 1

        grid[goal] = GOAL
        path = shortest_path(grid, start, goal)

        if path is not None and len(path) > 2:
            return grid, start, goal