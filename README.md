# DreamGrid

I built a tiny world model that learns to imagine before it acts.

The basic idea is like this. You show a neural network an 80x80 image of a grid. It learns what the colors mean. Then it learns to predict what happens when you move in a direction. And then instead of actually moving and seeing, it imagines moving and uses those imagined futures to decide what to do in the real environment.

85% success rate. Better than random or greedy but loses to the oracle that knows the true map. The model decides what to do purely by imagining, never peeking at what actually happens in the real world while planning.

## Quick version

Can you learn a world model well enough to plan without calling the real simulator? Yeah. The model hallucinates. Drifts over time. But spatial representations help enough that it works anyway.

## Try it

Web demo: https://yakshithk.github.io/DreamGrid/

Train it yourself:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YakshithK/DreamGrid/blob/master/notebooks/DreamGrid_Demo.ipynb)

### Real episode

![VQ-MPC episode](outputs/final/episodes/vq_mpc_episode_seed10001_h8_c1024.png)

### What it imagined

![VQ Imagination](outputs/final/imaginations/vq_imagination_seed10000_h24_c4096_k4.png)

So the planner samples a bunch of random action sequences, like 4096 of them. Each one is 24 steps long. It imagines what happens with each one using the learned dynamics model. Not the real world. Just the learned model. Then it scores each imagined future and picks the best one. The image above shows the top 4 candidates. You can see paths that get closer to the goal, avoid walls and hazards, stuff like that.

## The setup

Its a 10x10 grid. Rendered as an 80x80 image.

Blue square is the agent. Beige is floor. Black is walls. Red is hazards that kill you. Green is the goal.

You can move up, down, left, right, or do nothing. Each step costs 0.05 points. You die if you hit a hazard (lose 10 points). You win 10 points if you reach the goal. If you take more than 40 steps the episode ends.

Very simple environment. The hard part is that the model has to figure out what the colors mean from looking at them. Then predict how they change when you move. Then use those predictions to plan without actually seeing what happens in the real world.

## The vision model

I needed to turn an 80x80 image into something the dynamics model could predict.

I tried just compressing the whole image into one big vector first. That sucked. The model would forget where the agent was. It couldn't keep track of anything spatial. So that didn't work.

Instead I used a VQ-VAE. It learns a 10x10 grid of discrete codes. Each cell is one code. The model learned that code 3 means wall, code 7 means the agent is here, etc. It only needed 17 different codes total.

99.6% tile accuracy. 97.8% agent position. The spatial structure was the difference. Walls don't teleport. The agent moves one cell at a time. Preserving that in the representation made the whole thing work.

## The world model

This is the thing that actually predicts. You give it the current 10x10 grid of codes and an action and it predicts the next grid. Also predicts whether you hit something, whether you won, what your reward is.

It gets it right about 99% of the time. Agent position is 93% accurate. Collision detection is 99.9% accurate.

### How I trained it

Multiple objectives at once. Predict the next code, the tile type, agent position, collisions, episode end, and reward. Predicting just one step ahead doesn't work. The model needs to care about everything simultaneously or it forgets what matters.

When it plans it never touches the real environment. Pure imagination. If the predictions were garbage this wouldn't work. They're mediocre at worst.  

## Planning with imagination

Sample 4096 random action sequences, each 24 steps. Imagine what happens with each using the learned model. Score on reward, crashes, goal proximity, penalties for weird states. Pick the best. Execute the first action. Replan.

MPC. Standard in robotics but usually with simulators. Here the model is learned.

24 step horizon because shorter doesn't see far enough, longer predictions drift. Random sampling because gradient based is overkill. 4096 candidates work fine for 85% success.

## Baselines

I tested against three things.

Random just picks random actions. Its bad.

Greedy always moves closer to the goal. Sounds like it should work but it doesn't. Gets stuck on walls, can't avoid hazards coming. Gets like 47% success.

Oracle cheats and knows the actual map and uses shortest path. Gets 100%. Its the ceiling.

My model had to beat random and greedy. Obviously can't beat oracle.

## Results (100 episodes)

I tested on 100 grids. Same configuration throughout. 24 step horizon, 4096 candidates.

| Method | Success | Crashes | Timeouts | Avg Reward | Avg Steps |
|---|---:|---:|---:|---:|---:|
| Random | 6% | 50% die | 44% timeout | -11.67 | 27.67 |
| Greedy | 47% | 0% | 53% timeout | 3.52 | 23.98 |
| **VQ-MPC** | **85%** | 0% | 15% timeout | 7.31 | 13.78 |
| Oracle | 100% | 0% | 0% | 9.64 | 8.17 |

85 out of 100. Greedy is 47. Random is 6. Oracle knows the map so its 100.

The learned model still fails. Sometimes it imagines a future that never happens. Sometimes loops. Sometimes wasts too many steps. But 85% means it actually works for planning.

These 100 grids were the ones I used developing the planner so not a fresh test set. Just what I benchmarked against.

## Is it planning or memorizing

65.5% match rate with oracle on the first move. So 1/3 of the time it picks something different. Still reaches the goal 85% of the time.

## What failed

Global latent vector approach: compress 80x80 to one number, predict forward. Agents disappeared. Two agents sometimes. Agent in walls. After 8 steps the visualizations were gibberish.

66% agent detection. 60% position. 82% collisions. Can't plan when you're wrong 40% of the time about where the agent is.

Spatial dynamics fixed it. Predict each 10x10 cell separately. 99% detection. 99% position. 99.9% collisions.

The world is spatial. Preserving that in the representation changes everything. Final VQ model kept spatial structure but learns from RGB instead of hand labeled tiles.

## Failure modes (15% of the time)

Model drift. After 20 steps the errors stack. The planner optimizes for a goal that isn't there. Like being sure you win at chess but you hallucinated your opponent's position.

Local optima. 4096 random sequences sometimes all lean the same way. The planner commits, gets turned around, burns steps before realizing the path was wrong.

The done signal lies. 99% accurate but 1% says goal when the agent is 5 squares away. The planner goes there. Doesn't work.

Conservative paths. Safe but slow. Takes 25 steps instead of 8. Works. Wastes time.

Example timeout:

![VQ-MPC timeout](outputs/final/failures/vq_mpc_timeout_seed10000_h8_c1024.png)

Agent cycles through 4 moves. Never crashes. Never reaches goal. The model drifted and imagined a goal that doesn't exist. Planner kept returning to that phantom spot.

Model drift is the wall. 85% success but not robot-reliable.

## Why I built this

Most vision models classify. "Whats in this image?" They don't predict how things change. Can't answer "what if I move left?"

I wanted something that imagines. Takes an image, understands the layout, predicts what happens when you move, and plans from that. Not guessing. Actually thinking forward.

Wanted to build the whole pipeline myself. Environment generation. Vision model. Dynamics. Planner. See where it breaks.

The global latent approach broke everywhere. Agents vanished. Hallucinated multiple agents. Predictions that looked like noise. I was looking at visualizations at 3am just laughing at how wrong it was. But the failure was useful. Spatial structure matters. Grids should stay grids.

Built something that imagines okay. Not perfect. Good enough to navigate without the map.

## Run it

Use Colab for free GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YakshithK/DreamGrid/blob/master/notebooks/DreamGrid_Demo.ipynb)

Or locally:

```bash
pip install -r requirements.txt
```

In order:

```bash
# Generate training data
python -m scripts.generate_transitions

# Train vision model
python -m training.final.train_vqvae

# Train world model
python -m training.final.train_vq_dynamics
```

Then test it:

```bash
# Test 100 episodes
python -m eval.planners.evaluate_vq_planners \
  --episodes 100 --horizon 24 --candidates 4096

# See one episode
python -m eval.visualizations.visualize_vq_mpc_episode \
  --seed 10001 --horizon 24 --candidates 4096

# See what it imagined
python -m eval.visualizations.visualize_vq_imagination \
  --seed 10000 --horizon 24 --candidates 4096 --top_k 4
```

Takes a few minutes on GPU per step.

Can you learn a world model well enough to plan? Yeah. 85% success. Better than random or greedy. Worse than oracle. Sometimes times out.

The model sees a grid, predicts futures, navigates. No hand coded logic. No peeking at the simulator while planning. Just imagination.

Information survives quantization and spatial representation and forward prediction and actually gets used for planning. Its fragile but works.

Model drift breaks it after 20 steps. I don't have a fix. Maybe longer training or better loss design. Stopped here though.

Use A* if you need grid navigation in production. This is interesting if you care about world models or why spatial structure helps.

Code is there. Colab notebook ready. Visualizations are fun to watch. Run it if you want.
