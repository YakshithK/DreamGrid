from env.constants import GRID_SIZE, HAZARD, WALL, ACTION_TO_DELTA
from collections import deque

def in_bounds(pos):
    r, c = pos
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def is_walkable(grid, pos):
    r, c = pos
    return grid[r, c] != WALL and grid[r, c] != HAZARD 

def shortest_path(grid, start, goal):
    queue = deque([start])

    parent = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for action, delta in ACTION_TO_DELTA.items():
            if action == 4:
                continue

            dr, dc = delta
            nxt = (current[0] + dr, current[1] + dc)

            if not in_bounds(nxt) or not is_walkable(grid, nxt) or nxt in parent:
                continue

            parent[nxt] = current
            queue.append(nxt)

    if goal not in parent:
        return None
    
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path

def action_between(a, b):
    dr = b[0] - a[0]
    dc = b[1] - a[1]

    for action, delta in ACTION_TO_DELTA.items():
        if delta == (dr, dc):
            return action
        
    raise ValueError(f"No action between {a} and {b}")