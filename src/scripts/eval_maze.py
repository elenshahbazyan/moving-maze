
import os, sys, argparse, random, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
parser.add_argument("--episodes",   type=int,  default=100)
parser.add_argument("--render",     action="store_true")
parser.add_argument("--seed",       type=int,  default=42)
args = parser.parse_args()

DEVICE = torch.device("cpu")
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

ckpt_path = os.path.join(SCRIPT_DIR, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint

if not os.path.exists(ckpt_path):
    print(f"[Error] Checkpoint not found: {ckpt_path}")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location=DEVICE)
maze_data = ckpt.get("maze", None)

# Fallback: try loading maze.pkl from checkpoints folder
if maze_data is None:
    pkl = os.path.join(os.path.dirname(ckpt_path), "maze.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            maze_data = pickle.load(f)
        print(f"[Maze] Loaded from {pkl}")
    else:
        print(f"[Error] No maze data in checkpoint and no maze.pkl found.")
        print(f"        Train with train_maze.py first.")
        sys.exit(1)
else:
    print(f"[Maze] Loaded from checkpoint")

SIZE = maze_data["size"]
print(f"\n{'═'*60}")
print(f"  Eval  |  maze={SIZE}×{SIZE}  |  episodes={args.episodes}")
print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
print(f"  Maze: walls={len(maze_data['walls'])}  "
      f"gridlocks={len(maze_data['gridlocks'])}  "
      f"key_walls={len(maze_data['key_walls'])}")
print(f"{'═'*60}\n")

root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if root not in sys.path: sys.path.insert(0, root)
from src.environment.gridworld import GridWorldEnv

env = GridWorldEnv(
    size=SIZE, maze_type="multiple_paths",
    wall_update_freq=5,
    n_gridlocks=len(maze_data["gridlocks"]),
    n_key_walls=len(maze_data["key_walls"]),
    n_portal_pairs=0,
    render_mode="human" if args.render else None,
)

# Restore exact training maze
env.walls     = set(tuple(w) for w in maze_data["walls"])
env.gridlocks = set(tuple(g) for g in maze_data["gridlocks"])
env.key_walls = {
    tuple(int(x) for x in k.split(",")): v
    for k, v in maze_data["key_walls"].items()
}
env.free = {
    (x, y)
    for y in range(env.size)
    for x in range(env.size)
    if (x, y) not in env.walls
    and (x, y) not in env.gridlocks
    and (x, y) not in env.key_walls
}
print(f"[Env] Maze restored. free cells={len(env.free)}")

# Pre-load icons BEFORE first render so they appear correctly
if args.render:
    import pygame
    # asset_path is relative to gridworld.py location
    gw_dir      = os.path.dirname(os.path.abspath(
                    sys.modules["src.environment.gridworld"].__file__))
    asset_path  = os.path.normpath(os.path.join(gw_dir, "../../assets"))
    print(f"[Icons] Loading from: {asset_path}")
    try:
        env.agent_icon          = pygame.image.load(os.path.join(asset_path, "agent_icon.png")).convert_alpha()
        env.target_icon         = pygame.image.load(os.path.join(asset_path, "targets_icon.jpg")).convert()
        env.key_icon            = pygame.image.load(os.path.join(asset_path, "door.png")).convert_alpha()
        env.portal_icon_closer  = pygame.image.load(os.path.join(asset_path, "teleport_closer.png")).convert_alpha()
        env.portal_icon_farther = pygame.image.load(os.path.join(asset_path, "teleport_farther.png")).convert_alpha()
        print("[Icons] All icons loaded successfully")
    except Exception as e:
        print(f"[Icons] Warning: could not load icons ({e}) — using colored squares")

def reveal_full(env):
    for y in range(env.size):
        for x in range(env.size):
            env.known_map[y, x] = env._encode_cell(x, y)

def env_reset(env):
    env.reset()
    reveal_full(env)
    return env._get_obs()

def env_step(env, action):
    _, env_r, done, trunc, info = env.step(action)
    reveal_full(env)
    return env._get_obs(), env_r, done, trunc, info

def mdist(raw):
    a = np.array(raw["agent"],  float)
    t = np.array(raw["target"], float)
    return float(np.abs(a - t).sum())

def keywall_ahead(env, raw):
    ax, ay = int(raw["agent"][0]), int(raw["agent"][1])
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        pos = (ax+dx, ay+dy)
        if pos in env.key_walls and pos not in env.free:
            return True
    return False

OBS_DIM = 7 + 49 + SIZE * SIZE
BUDGET  = SIZE * SIZE * 2

def encode(raw):
    s  = max(SIZE - 1, 1)
    ag = np.array(raw["agent"],  np.float32) / s
    tg = np.array(raw["target"], np.float32) / s
    dl = (tg * s - ag * s) / s
    d  = np.array([float(np.abs(dl).sum()) / 2.0], np.float32)
    lo = np.array(raw["local"],  np.float32).flatten() / 6.0
    km = np.array(raw["known_map"], np.float32).flatten() / 6.0
    return torch.from_numpy(np.concatenate([ag, tg, dl, d, lo, km]).astype(np.float32))

class DuelingDQN(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_act=5, hidden=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
        )
        h = hidden // 2
        self.V = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, 1))
        self.A = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, n_act))

    def forward(self, x):
        f = self.shared(x)
        v = self.V(f); a = self.A(f)
        return v + a - a.mean(1, keepdim=True)

# Detect actual input dim from saved weights
actual_dim = ckpt["Q"]["shared.0.weight"].shape[1]
if actual_dim != OBS_DIM:
    print(f"[Warn] checkpoint in_dim={actual_dim}, expected {OBS_DIM}")
    print(f"       Using checkpoint's dimension.")
    OBS_DIM = actual_dim
    BUDGET  = SIZE * SIZE * 2

model = DuelingDQN(obs_dim=actual_dim)
model.load_state_dict(ckpt["Q"])
model.eval()
print(f"[Model] in_dim={actual_dim}  t={ckpt.get('t','?')}  eps={ckpt.get('eps','?'):.3f}\n")

print(f"[Running {args.episodes} episodes...]\n")
results = []

for ep in range(1, args.episodes + 1):
    raw = env_reset(env)
    success = False
    ep_steps = ep_walls = 0

    for _ in range(BUDGET):
        at_kw = keywall_ahead(env, raw)
        s = encode(raw).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = model(s)[0].clone()
        if not at_kw:
            q[4] -= 0.3   # discourage waiting when no key wall nearby
        action = int(q.argmax())

        raw, _, done, trunc, info = env_step(env, action)
        ep_steps += 1
        if info.get("hit_wall", False): ep_walls += 1

        if done:
            success = True
            break
        if trunc:
            break

    final_d = mdist(raw)
    results.append(dict(ep=ep, success=success, steps=ep_steps,
                        dist=final_d, walls=ep_walls))

    marker = "✓ GOAL" if success else "✗ fail"
    print(f"  ep {ep:4d}/{args.episodes}  {marker}  "
          f"steps={ep_steps:4d}  dist={final_d:5.1f}  walls={ep_walls}")

env.close()

n     = len(results)
succ  = [r for r in results if r["success"]]
sr    = len(succ) / n * 100
avg_d = np.mean([r["dist"]  for r in results])
avg_s = np.mean([r["steps"] for r in results])
avg_w = np.mean([r["walls"] for r in results])

print(f"\n{'═'*60}")
print(f"  RESULTS  —  {n} episodes  |  {SIZE}×{SIZE} maze")
print(f"{'═'*60}")
print(f"  ✓  Success rate : {sr:.1f}%  ({len(succ)}/{n})")
print(f"  ─  Avg steps    : {avg_s:.1f}")
print(f"  ─  Avg dist     : {avg_d:.2f}")
print(f"  ─  Avg walls    : {avg_w:.1f}")
if succ:
    print(f"  ─  Avg steps(✓) : {np.mean([r['steps'] for r in succ]):.1f}")
print(f"{'═'*60}\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Eval — {SIZE}×{SIZE} — SR={sr:.1f}% ({len(succ)}/{n})", fontsize=13)

eps_ax = [r["ep"]      for r in results]
dists  = [r["dist"]    for r in results]
steps  = [r["steps"]   for r in results]
colors = ["green" if r["success"] else "red" for r in results]

axes[0].bar(eps_ax, [100 if r["success"] else 0 for r in results],
            color=colors, alpha=0.5, width=1)
axes[0].axhline(sr, color="gold", lw=2, ls="--", label=f"Mean={sr:.1f}%")
axes[0].set_title("Success per Episode"); axes[0].legend(fontsize=9)

axes[1].scatter(eps_ax, dists, c=colors, alpha=0.6, s=15)
axes[1].axhline(0, color="green", lw=1.5, ls="--", label="Goal")
axes[1].set_title("Final Distance"); axes[1].legend(fontsize=9)

axes[2].scatter(eps_ax, steps, c=colors, alpha=0.6, s=15)
axes[2].axhline(avg_s, color="gold", lw=2, ls="--", label=f"Mean={avg_s:.0f}")
axes[2].set_title("Steps per Episode"); axes[2].legend(fontsize=9)

plt.tight_layout()
stem     = os.path.splitext(os.path.basename(ckpt_path))[0]
out_path = os.path.join(os.path.dirname(ckpt_path), f"eval_{stem}.png")
plt.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"[Plot] → {out_path}")
print(f"  Done.\n")