
"""

Headless inspector that checks and visualizes the GridWorldEnv visibility
(local + known_map). Saves all plots to results/plots/ so you don't need to
modify the environment code.

Usage (from project root):
    python src/scripts/debug_visibility.py

If import fails, run with PYTHONPATH set to project root, e.g.:
    PYTHONPATH=. python src/scripts/debug_visibility.py

This script intentionally *does not* modify gridworld_maze.py — it only
imports and calls the public API (and a couple of internal helpers for
convenience when running tests).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root / src is on path so imports work when run from project root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import the environment (works if your package layout is unchanged)
try:
    from src.environment.gridworld_maze import GridWorldEnv
except Exception as e:
    # second attempt: maybe user runs differently; try a fallback import path
    try:
        from src.environment.gridworld_maze import GridWorldEnv
    except Exception:
        print("Failed to import GridWorldEnv. Make sure to run this script from the project root or set PYTHONPATH.")
        raise

# ---------- helpers ----------

def ascii_patch(arr):
    key = {0: '?', 1: '.', 2: '#', 3: 'G', 4: 'K', 5: 'P', 6: 'T'}
    out = []
    for row in arr:
        out.append(''.join(key.get(int(v), '?') for v in row))
    return "\n".join(out)


def show_known_map(km, save_path=None, title="known_map"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(km, interpolation='nearest', vmin=0, vmax=6)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


class VisibilityInspector:
    def __init__(self, env: GridWorldEnv, out_dir="results/plots"):
        self.env = env
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def reset_and_print(self):
        """Reset env and print basic info + save known_map image."""
        obs, _ = self.env.reset()
        print("Agent:", obs["agent"], "Target:", obs["target"])
        print("Local (center is agent):")
        print(ascii_patch(obs["local"]))

        fname = os.path.join(self.out_dir, "known_map_reset.png")
        show_known_map(obs["known_map"], save_path=fname, title="known_map (reset)")
        print("Saved:", fname)
        return obs

    def step_and_show(self, action, step_i):
        res = self.env.step(action)
        # support both (obs, r, done, trunc, info) and (obs, r, done, info)
        if len(res) == 5:
            obs, reward, done, trunc, info = res
        else:
            obs, reward, done, info = res
            trunc = False
        print(f"Step {step_i} action={action} agent={obs['agent']} reward={reward} done={done}")
        print(ascii_patch(obs["local"]))
        fname = os.path.join(self.out_dir, f"known_map_step{step_i}.png")
        show_known_map(obs["known_map"], save_path=fname, title=f"known_map step {step_i}")
        print("Saved:", fname)
        return obs, done

    def check_local_center_correct(self):
        obs, _ = self.env.reset()
        ax, ay = int(obs["agent"][0]), int(obs["agent"][1])
        center_val = obs["local"][self.env.vision_radius, self.env.vision_radius]
        try:
            actual = self.env._encode_cell(ax, ay)
        except Exception:
            actual = None
        print("local center:", center_val, "encode_cell:", actual)
        return center_val == actual

    def check_staleness_for(self, cell, n_updates=50):
        """Reveal the cell by moving agent there, then run wall updates repeatedly
        to see whether known_map becomes stale relative to current encode.
        """
        x, y = cell
        print(f"Temporarily placing agent at {cell} to reveal it (for test)")
        # temporarily set agent location and reveal (this uses internal helper)
        self.env._agent_location = np.array([x, y])
        try:
            self.env._reveal_from(x, y)
        except Exception:
            print("_reveal_from not available; skipping staleness test")
            return None

        known_before = self.env.known_map[y, x]
        actual_before = self.env._encode_cell(x, y)
        print("revealed known_map:", known_before, "actual:", actual_before)

        # call updates many times to try and flip state (randomized)
        for i in range(n_updates):
            self.env._update_moving_walls()

        known_after = self.env.known_map[y, x]
        actual_after = self.env._encode_cell(x, y)
        print("after updates known_map:", known_after, "actual:", actual_after)
        return (known_before, actual_before, known_after, actual_after)


# ---------- main ----------

def main():
    env = GridWorldEnv(render_mode=None, size=20)
    inspector = VisibilityInspector(env, out_dir=os.path.join(PROJECT_ROOT, "results/plots"))

    print("Sanity check: does local center match encode_cell?")
    ok = inspector.check_local_center_correct()
    print("OK:", ok)

    obs = inspector.reset_and_print()

    # sample actions - you can change these or randomize
    actions = [0, 0, 1, 0, 2, 3]
    for i, a in enumerate(actions, start=1):
        obs, done = inspector.step_and_show(a, i)
        if done:
            print("Reached goal at step", i)
            break

    # staleness test (if any dynamic walls exist)
    if getattr(env, 'gridlocks', None):
        gcell = next(iter(env.gridlocks))
        print("Running staleness test for gridlock cell:", gcell)
        res = inspector.check_staleness_for(gcell, n_updates=100)
        print("staleness result:", res)
    else:
        print("No gridlocks present to test staleness.")

    print("All plots saved to:", os.path.join(PROJECT_ROOT, "results/plots"))


if __name__ == '__main__':
    main()
