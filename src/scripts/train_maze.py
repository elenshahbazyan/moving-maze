"""
train_maze.py  —  Train DDQN on one fixed 30×30 maze
USAGE:
  python train_maze.py
  python train_maze.py --episodes 6000
"""

import os, sys, math, random, pickle, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

try:
    from tqdm import tqdm; TQDM = True
except ImportError:
    TQDM = False

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=5000)
parser.add_argument("--size",     type=int, default=30)
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

SIZE        = args.size
SEED        = args.seed
N_EPISODES  = args.episodes
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR    = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
MAZE_PKL    = os.path.join(CKPT_DIR, "maze.pkl")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[Config] size={SIZE}  episodes={N_EPISODES}  device={DEVICE}  seed={SEED}")

root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if root not in sys.path: sys.path.insert(0, root)
from src.environment.gridworld import GridWorldEnv

def make_env():
    return GridWorldEnv(
        size=SIZE, maze_type="multiple_paths",
        wall_update_freq=5,
        n_gridlocks=15, n_key_walls=8,
        n_portal_pairs=0,
    )

def save_maze(env, path):
    data = {
        "walls":     [list(w) for w in env.walls],
        "gridlocks": [list(g) for g in env.gridlocks],
        "key_walls": {f"{k[0]},{k[1]}": v for k, v in env.key_walls.items()},
        "size":      env.size,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[Maze] Saved → {path}")
    print(f"       walls={len(data['walls'])}  gridlocks={len(data['gridlocks'])}  key_walls={len(data['key_walls'])}")
    return data

def restore_maze(env, data):
    env.walls     = set(tuple(w) for w in data["walls"])
    env.gridlocks = set(tuple(g) for g in data["gridlocks"])
    env.key_walls = {
        tuple(int(x) for x in k.split(",")): v
        for k, v in data["key_walls"].items()
    }
    env.free = {
        (x, y)
        for y in range(env.size)
        for x in range(env.size)
        if (x, y) not in env.walls
        and (x, y) not in env.gridlocks
        and (x, y) not in env.key_walls
    }

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

# Vector layout (fixed for SIZE):
#   ag[2] + tg[2] + delta[2] + dist[1] + local[49] + known_map[SIZE*SIZE]
OBS_DIM = 7 + 49 + SIZE * SIZE

def encode(raw) -> torch.Tensor:
    s = max(SIZE - 1, 1)
    ag = np.array(raw["agent"],  np.float32) / s
    tg = np.array(raw["target"], np.float32) / s
    dl = (tg * s - ag * s) / s          # normalized delta
    d  = np.array([float(np.abs(dl).sum()) / 2.0], np.float32)
    lo = np.array(raw["local"],  np.float32).flatten() / 6.0
    km = np.array(raw["known_map"], np.float32).flatten() / 6.0
    return torch.from_numpy(np.concatenate([ag, tg, dl, d, lo, km]).astype(np.float32))

def mdist(raw):
    a = np.array(raw["agent"],  float)
    t = np.array(raw["target"], float)
    return float(np.abs(a - t).sum())

def keywall_ahead(env, raw):
    """Returns True if any of the 4 neighbours is a closed key wall."""
    ax, ay = int(raw["agent"][0]), int(raw["agent"][1])
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        pos = (ax+dx, ay+dy)
        if pos in env.key_walls and pos not in env.free:
            return True
    return False

BUDGET = SIZE * SIZE * 2   # max steps per episode

MAX_DIST = 2.0 * (SIZE - 1)

class Shaper:
    def __init__(self):
        self._prev = None

    def reset(self, raw):
        self._prev = -mdist(raw) / MAX_DIST

    def __call__(self, raw, raw_n, env_r, done, hit_wall, action, waiting_at_keywall):
        phi_s  = self._prev
        phi_ns = -mdist(raw_n) / MAX_DIST
        self._prev = phi_ns

        r = 0.99 * phi_ns - phi_s   # potential-based shaping
        r -= 0.005                   # small step penalty

        if done:
            r += 100.0               # big goal reward
        if hit_wall:
            r -= 0.3                 # wall penalty
        if action == 4:
            if waiting_at_keywall:
                r += 0.15            # reward smart waiting at closed key wall
            else:
                r -= 0.1             # penalise pointless waiting elsewhere

        return float(r)

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

class ReplayBuffer:
    def __init__(self, cap=150_000):
        self.cap = cap
        self.buf = []
        self.ptr = 0

    def push(self, *transition):
        if len(self.buf) < self.cap:
            self.buf.append(transition)
        else:
            self.buf[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.cap

    def sample(self, n):
        return random.sample(self.buf, min(n, len(self.buf)))

    def __len__(self):
        return len(self.buf)

class Agent:
    def __init__(self):
        self.Q  = DuelingDQN().to(DEVICE)
        self.Qt = DuelingDQN().to(DEVICE)
        self.Qt.load_state_dict(self.Q.state_dict())
        self.Qt.eval()

        self.opt = optim.AdamW(self.Q.parameters(), lr=3e-4, weight_decay=1e-5)
        self.buf = ReplayBuffer(150_000)

        self.gamma    = 0.99
        self.batch    = 256
        self.tgt_upd  = 400
        self.t        = 0

        # epsilon schedule
        self.eps      = 1.0
        self.eps_min  = 0.05
        self.eps_dec  = 0.9995   # multiplied every episode

        self.losses = []
        self.qvals  = []

    def act(self, raw, greedy=False, waiting_at_keywall=False):
        if not greedy and random.random() < self.eps:
            # Allow wait action during exploration only when at key wall
            n_actions = 5 if waiting_at_keywall else 4
            return random.randint(0, n_actions - 1)
        s = encode(raw).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = self.Q(s)[0].clone()
        if not waiting_at_keywall:
            q[4] -= 0.3   # discourage waiting when no key wall nearby
        return int(q.argmax())

    def push(self, raw, a, r, raw_n, done):
        s  = encode(raw).numpy()
        ns = encode(raw_n).numpy()
        self.buf.push(s, int(a), float(r), ns, float(done))

    def learn(self):
        if len(self.buf) < max(self.batch, 2000):
            return
        batch = self.buf.sample(self.batch)
        S, A, R, NS, D = zip(*batch)
        S  = torch.FloatTensor(np.array(S)).to(DEVICE)
        A  = torch.LongTensor(A).to(DEVICE)
        R  = torch.FloatTensor(R).to(DEVICE)
        NS = torch.FloatTensor(np.array(NS)).to(DEVICE)
        D  = torch.FloatTensor(D).to(DEVICE)

        with torch.no_grad():
            na  = self.Q(NS).argmax(1)
            nq  = self.Qt(NS).gather(1, na.unsqueeze(1)).squeeze(1)
            tgt = R + self.gamma * nq * (1 - D)

        cq   = self.Q(S).gather(1, A.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(cq, tgt)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q.parameters(), 10.0)
        self.opt.step()

        self.t += 1
        if self.t % self.tgt_upd == 0:
            self.Qt.load_state_dict(self.Q.state_dict())

        self.losses.append(loss.item())
        self.qvals.append(cq.mean().item())

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_dec)

    def save(self, path, maze_data=None):
        tmp = path + ".tmp"
        data = {
            "Q":   self.Q.state_dict(),
            "Qt":  self.Qt.state_dict(),
            "opt": self.opt.state_dict(),
            "t":   self.t,
            "eps": self.eps,
        }
        if maze_data is not None:
            data["maze"] = maze_data
        torch.save(data, tmp)
        if os.path.exists(path): os.remove(path)
        os.rename(tmp, path)

    def load(self, path):
        c = torch.load(path, map_location=DEVICE)
        self.Q.load_state_dict(c["Q"])
        self.Qt.load_state_dict(c["Qt"])
        if "opt" in c: self.opt.load_state_dict(c["opt"])
        self.t   = c.get("t", 0)
        self.eps = c.get("eps", self.eps_min)
        return c.get("maze", None)

def evaluate(agent, env, n=20):
    success = 0
    for _ in range(n):
        raw = env_reset(env)
        for _ in range(BUDGET):
            a = agent.act(raw, greedy=True)
            raw, _, done, trunc, _ = env_step(env, a)
            if done:  success += 1; break
            if trunc: break
    return success / n

def plot(ep_S, ep_D, ep_R, losses, qvals, goals, path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Train — {SIZE}×{SIZE} — goals={goals}", fontsize=14)

    n   = len(ep_S)
    eps = np.arange(1, n + 1)
    S   = np.array(ep_S, float)

    def sm(arr, w=50):
        arr = np.array(arr, float)
        if len(arr) < w: return np.arange(len(arr)), arr
        return np.arange(w - 1, len(arr)), np.convolve(arr, np.ones(w) / w, "valid")

    ax = axes[0, 0]
    ax.bar(eps, S * 100, width=1, color=["green" if s else "red" for s in ep_S], alpha=0.4)
    if n >= 100:
        xs, ys = sm(S, 100)
        ax.plot(xs, ys * 100, "b-", lw=2, label="Rolling SR (100ep)")
    ax.set_title("Success Rate"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    xs, ys = sm(np.array(ep_D, float), 50)
    ax.plot(xs, ys, "orange", lw=2)
    ax.axhline(0, color="green", ls="--", lw=1.5, label="Goal")
    ax.set_title("Final Distance"); ax.legend(fontsize=8)

    ax = axes[0, 2]
    if losses:
        xs, ys = sm(losses, min(500, len(losses) // 4 + 1))
        ax.semilogy(xs, ys, "red", lw=2)
    ax.set_title("TD Loss")

    ax = axes[1, 0]
    r_arr = np.array(ep_R, float)
    ax.fill_between(eps, r_arr, alpha=0.15, color="steelblue")
    if len(r_arr) >= 50:
        xsr, ysr = sm(r_arr, 50)
        ax.plot(xsr, ysr, color="steelblue", lw=2, label="Smoothed (50ep)")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("Episode Reward"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    if qvals:
        xs, ys = sm(qvals, min(500, len(qvals) // 4 + 1))
        ax.plot(xs, ys, "gold", lw=2)
    ax.set_title("Mean Q-value")

    ax = axes[1, 2]; ax.axis("off")
    ax.text(0.1, 0.9,
            f"Size: {SIZE}×{SIZE}\nEpisodes: {n}\nGoals: {goals}\n"
            f"Budget: {BUDGET} steps\nSeed: {SEED}",
            transform=ax.transAxes, fontsize=12, va="top", family="monospace")

    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Plot] → {path}")

def train():
    # Create env and fix maze
    env = make_env()
    maze_data = save_maze(env, MAZE_PKL)
    # Restore to make sure env.free is consistent
    restore_maze(env, maze_data)

    agent = Agent()

    ep_S, ep_D, ep_R = [], [], []
    rS = deque(maxlen=100)
    best_sr  = -1.0
    goals    = 0
    learn_every = 2
    gsteps   = 0

    print(f"\n[Start] obs_dim={OBS_DIM}  budget={BUDGET}")
    print(f"[Maze]  walls={len(env.walls)}  gridlocks={len(env.gridlocks)}  key_walls={len(env.key_walls)}\n")

    bar = tqdm(total=N_EPISODES, unit="ep", dynamic_ncols=True) if TQDM else None

    for ep in range(1, N_EPISODES + 1):
        raw    = env_reset(env)
        shaper = Shaper()
        shaper.reset(raw)

        ep_r = 0.0
        success = False

        for _ in range(BUDGET):
            at_kw  = keywall_ahead(env, raw)
            action = agent.act(raw, waiting_at_keywall=at_kw)
            raw_n, env_r, done, trunc, info = env_step(env, action)

            hit = bool(info.get("hit_wall", False))
            r   = shaper(raw, raw_n, env_r, done, hit, action, at_kw)

            agent.push(raw, action, r, raw_n, done or trunc)
            raw   = raw_n
            ep_r += r
            gsteps += 1

            if gsteps % learn_every == 0:
                agent.learn()

            if done:
                success = True
                goals  += 1
                break
            if trunc:
                break

        agent.decay_eps()
        final_d = mdist(raw)
        ep_S.append(int(success))
        ep_D.append(final_d)
        ep_R.append(ep_r)
        rS.append(int(success))

        sr = np.mean(rS) * 100

        if bar:
            bar.set_postfix(SR=f"{sr:.0f}%", d=f"{final_d:.0f}",
                            ε=f"{agent.eps:.3f}", goals=goals,
                            res="✓" if success else "✗")
            bar.update(1)
        elif ep % 100 == 0:
            print(f"ep={ep:4d}  SR={sr:.1f}%  d={final_d:.0f}  ε={agent.eps:.3f}  goals={goals}")

        # Eval and checkpoint every 200 episodes
        if ep % 200 == 0:
            esr = evaluate(agent, env, n=20)
            msg = f"  ► Eval ep{ep}: SR={esr*100:.1f}%"
            (tqdm.write if TQDM else print)(msg)
            if esr > best_sr:
                best_sr = esr
                agent.save(os.path.join(CKPT_DIR, "best_model.pt"), maze_data)
                (tqdm.write if TQDM else print)(f"    ★ New best! Saved.")

        # Periodic checkpoint every 500 episodes
        if ep % 500 == 0:
            p = os.path.join(CKPT_DIR, f"ckpt_ep{ep:05d}.pt")
            agent.save(p, maze_data)
            # keep only 3 most recent
            old = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_ep*.pt")))
            for f in old[:-3]:
                try: os.remove(f)
                except: pass

    if bar: bar.close()

    # Final save
    agent.save(os.path.join(CKPT_DIR, "final_model.pt"), maze_data)
    env.close()

    plot(ep_S, ep_D, ep_R, agent.losses, agent.qvals, goals,
         os.path.join(CKPT_DIR, "training_results.png"))

    print(f"  Done | goals={goals}")
    print(f"  Last 100ep SR : {np.mean(ep_S[-100:])*100:.1f}%")
    print(f"  Best eval SR  : {best_sr*100:.1f}%")
    print(f"  Checkpoints   : {CKPT_DIR}")
    print(f"{'═'*60}")

if __name__ == "__main__":
    train()