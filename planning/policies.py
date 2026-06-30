import random

from env.constants import ACTION_TO_DELTA, NUM_ACTIONS
from env.pathfinding import action_between, shortest_path


class RandomPolicy:
    def act(self, obs, env):
        return random.randrange(NUM_ACTIONS), {}


class GreedyPolicy:
    def act(self, obs, env):
        from env.constants import WALL, HAZARD

        best_action = 4
        best_dist = manhattan(env.agent_pos, env.goal_pos)

        for action, delta in ACTION_TO_DELTA.items():
            nr = env.agent_pos[0] + delta[0]
            nc = env.agent_pos[1] + delta[1]
            pos = (nr, nc)

            if not env._in_bounds(pos):
                continue

            if env.grid[pos] in (WALL, HAZARD):
                continue

            dist = manhattan(pos, env.goal_pos)

            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action, {}
    

class OracleShortestPathPolicy:
    def act(self, obs, env):
        path = shortest_path(env.grid, env.agent_pos, env.goal_pos)
        
        if path is None or len(path) < 2:
            return random.randint(0, NUM_ACTIONS - 1), {"path_found": False}
        
        action = action_between(path[0], path[1])
        return action, {"path_found": True}


class PlannerPolicy:
    def __init__(self, planner):
        self.planner = planner

    def act(self, obs, env):
        return self.planner.plan(obs)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])