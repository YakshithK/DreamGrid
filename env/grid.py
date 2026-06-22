import numpy as np

from env.constants import GRID_SIZE, FLOOR, HAZARD, WALL, GOAL, MAX_STEPS, ACTION_TO_DELTA
from env.generator import generate_map
from env.render import render_grid

class RescueGridEnv:
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        map_seed = int(self.rng_integers(0, 1_000_000_000))
        self.grid, self.agent_pos, self.goal_pos = generate_map(seed=map_seed)
        self.steps = 0
        self.done = False

        return self.render()
    

    def step(self, action):
        if self.done:
            return self.render(), 0.0, True, {
                "collision": False,
                "success": False,
                "agent_pos": self.agent_pos,
                "goal_pos": self.goal_pos
            }
        
        self.steps += 1
        collision = False
        success = False

        dr, dc = ACTION_TO_DELTA[action]
        current_r, current_c = self.agent_pos
        next_pos = (current_r + dr, current_c + dc)

        reward = -0.05

        if not self._in_bounds(next_pos):
            next_pos =  self.agent_pos
            collision = True
            reward = -1.0
        else:
            tile = self.grid[next_pos]

            if tile == WALL:
                next_pos = self.agent_pos
                collision = True
                reward = -1.0
            elif tile == HAZARD:
                self.agent_pos = next_pos
                self.done = True
                reward = -10.0
            elif tile == GOAL:
                self.agent_pos = next_pos
                self.done = True
                success = True
                reward = 10.0
            else:
                self.agent_pos = next_pos

        if self.steps >= MAX_STEPS:
            self.done = True

        info = {
            "collision": collision,
            "success": success,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "steps": self.steps
        }

        return self.render(), reward, self.done, info
    

    def render(self):
        return render_grid(self.grid, self.agent_pos)
    
    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]