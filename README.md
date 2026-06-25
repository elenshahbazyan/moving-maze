# Dynamic Maze Navigation: From Tabular SARSA to Deep Dueling DDQN

A reinforcement learning agent trained to navigate a large, partially dynamic maze toward a fixed goal - implemented and compared across two algorithmic generations: a tabular SARSA baseline, followed by a Dueling Double DQN (DDQN) agent built to handle the scale and complexity the tabular approach could not.

## Demo

![Agent reaching the goal](./30x30.mp4)

*The trained agent navigating the maze and reaching the target.*

## Overview

The environment is not a static maze with a single fixed solution path. Built on top of a DFS-generated maze skeleton, two additional obstacle types make the layout **non-stationary**:

- **Gridlocks** - cells that randomly toggle between open and blocked at a fixed update interval.
- **Key walls** - cells with their own individual toggle probability, also updated on that same interval.

This turns the task from classic shortest-path search into a timing problem: the optimal policy isn't just "find a path around the walls," it's "find a path *and* know when to wait for a wall to open." A dedicated **WAIT action** is included in the action space specifically to support this — the agent has to learn that waiting is only valuable adjacent to a closed key wall, and wasteful everywhere else.

## Two-Stage Approach

### Stage 1 - Tabular SARSA (baseline)
An on-policy SARSA agent operating over a discretized state space. To account for the cyclic nature of the wall toggling without exploding the state space, the state is encoded as a function of agent position **and** step count modulo a fixed time window — letting the table capture "where am I, and where are we in the wall-toggle cycle" without needing a continuous representation. Trained with a linearly decaying epsilon-greedy policy and evaluated periodically with a greedy rollout; the best-performing Q-table is checkpointed independently of the final one.

This approach is simple, fully on-policy (safer in a stochastic environment, since it learns the value of the policy actually being followed rather than an idealized greedy one), and serves as an interpretable benchmark — but tabular state representation scales poorly as the grid grows and as the time window needed to capture wall dynamics widens.

### Stage 2 - Dueling Double DQN
To scale to a larger grid with denser dynamic obstacles, the tabular Q-table was replaced with a Dueling DDQN operating on a continuous feature vector (normalized agent/target position, relative delta, Manhattan distance, a local neighborhood view, and the known map of the maze), processed by a network that separately estimates state-value and action-advantage before recombining them.

Key components:
- **Double DQN target** - action selection from the online network, value estimation from a periodically-synced target network, reducing the overestimation bias of vanilla DQN.
- **Experience replay** - a 150k-capacity buffer decorrelating consecutive transitions for more stable gradient updates.
- **Reward shaping** - a potential-based shaping term (using Manhattan distance to goal, which preserves the optimal policy) on top of the sparse environment reward, plus a targeted bonus/penalty for the WAIT action depending on whether the agent is actually adjacent to a closed key wall — explicitly teaching the "smart waiting" behavior the SARSA baseline had to discover from a much coarser signal.

## Engineering Highlights

**Fixed-maze protocol** - the maze (walls, gridlock and key-wall placement) is generated once per run, serialized, and reused for the entire training session — and can be restored later for evaluation — so results reflect learning progress on a consistent layout rather than a constantly shifting target.

**Evaluation & checkpointing** - both agents run periodic greedy-policy evaluation episodes during training; a checkpoint is saved whenever evaluation success rate improves, alongside rolling periodic checkpoints with automatic pruning of older ones.

**Diagnostics** - automated training dashboards covering success rate, distance-to-goal, TD loss, episode reward, and Q-value trends, generated directly from training logs.

**Video capture** - evaluation rollouts can be rendered and recorded as the agent solves the maze, providing a direct visual artifact of policy quality rather than relying on metrics alone.

## Tech Stack

- **Python / NumPy** - core implementation, tabular SARSA
- **PyTorch** - Dueling DDQN network and training loop
- **Gymnasium** - custom maze environment, `gym.Env` interface
- **Pygame** - rendering and video capture
- **Matplotlib** - training diagnostics
