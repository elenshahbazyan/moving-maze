import os
import sys
import random
import time

# Ensure project root (two levels up from src/scripts) is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try common module names (adjust if your file name is different)
try:
    from src.environment.gridworld_maze import GridWorldEnv
except Exception:
    try:
        from src.environment.gridworld_maze import GridWorldEnv
    except Exception as e:
        # Helpful error explaining what's expected
        raise ImportError(
            "Could not import GridWorldEnv. Make sure your file is at "
            "src/environment/gridworld_maze.py or src/environment/gridworld_env.py "
            "and contains class GridWorldEnv."
        ) from e

def run_random():
    env = GridWorldEnv(render_mode="human", size=50)
    obs, info = env.reset()
    running = True
    try:
        for _ in range(10000):
            import pygame
            # keep window responsive
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
            if not running:
                break

            action = random.randint(0, env.action_space.n - 1)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                print("Episode finished:", done, truncated)
                break

            time.sleep(0.05)
    finally:
        env.close()

if __name__ == "__main__":
    run_random()
