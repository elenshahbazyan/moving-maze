"""
eval_ddqn_v3.py — Eval script matching train_ddqn_v3.py exactly
================================================================
USAGE (from src/scripts folder):
  python eval_ddqn_v3.py --checkpoint checkpoints_v3/goal_reached_checkpoints/goal_ep02000_total0050_sz20.pt --size 20
  python eval_ddqn_v3.py --checkpoint checkpoints_v3/best_model.pt --size 20 --render
"""

import os, sys, argparse, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--size",     type=int, default=20)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--render",   action="store_true")
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

EVAL_SIZE  = args.size
FINAL_SIZE = 20   # must match training FINAL_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

print(f"\n{'═'*64}")
print(f"  DDQN Eval v3  (matches train_ddqn_v3.py)")
print(f"  Checkpoint : {os.path.basename(args.checkpoint)}")
print(f"  Maze size  : {EVAL_SIZE}×{EVAL_SIZE}  |  Network: {FINAL_SIZE}×{FINAL_SIZE}")
print(f"  Episodes   : {args.episodes}  |  Device: {DEVICE}")
print(f"{'═'*64}\n")

# ══════════════════════════════════════════════════════════════════
#  ENV
# ══════════════════════════════════════════════════════════════════
def load_env_class():
    root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    if root not in sys.path: sys.path.insert(0, root)
    from src.environment.gridworld import GridWorldEnv
    print("[Env] loaded")
    return GridWorldEnv

def make_env(size, render=False, maze_data=None):
    GridWorldEnv = load_env_class()
    env = GridWorldEnv(
        size=size, maze_type="multiple_paths",
        wall_update_freq=5, gridlock_ratio=0.15,
        keywall_ratio=0.10, n_portal_pairs=0,
        render_mode="human" if render else None,
    )
    if maze_data is not None:
        # Restore exact maze from checkpoint so eval uses same maze as training
        env.walls     = set(tuple(w) for w in maze_data["walls"])
        env.gridlocks = set(tuple(g) for g in maze_data["gridlocks"])
        env.key_walls = {
            tuple(int(x) for x in k.strip("()").split(",")): v
            for k, v in maze_data["key_walls"].items()
        }
        env.free = {
            (x, y)
            for y in range(env.size)
            for x in range(env.size)
            if (x,y) not in env.walls
            and (x,y) not in env.gridlocks
            and (x,y) not in env.key_walls
        }
        print(f"[Maze] Restored from checkpoint: {len(env.walls)} walls, "
              f"{len(env.gridlocks)} gridlocks, {len(env.key_walls)} key_walls")
    else:
        print("[Maze] WARNING: no maze in checkpoint — using a fresh random maze.")
        print("       This agent was trained on one fixed maze; results may be poor.")
    return env

def reveal_full_map(env):
    for y in range(env.size):
        for x in range(env.size):
            env.known_map[y, x] = env._encode_cell(x, y)

def full_obs(env):
    reveal_full_map(env); return env._get_obs()

def env_reset(env):
    env.reset(); return full_obs(env)

def env_step(env, action):
    _, env_r, done, trunc, info = env.step(action)
    return full_obs(env), env_r, done, trunc, info

# ══════════════════════════════════════════════════════════════════
#  OBS
# ══════════════════════════════════════════════════════════════════
def unpack(raw, size):
    obs = {}
    for k in ("agent","agent_pos","agent_location"):
        if k in raw: obs["agent"] = np.array(raw[k],np.int32).flatten()[:2]; break
    obs.setdefault("agent", np.zeros(2,np.int32))
    for k in ("target","target_pos","goal","target_location"):
        if k in raw: obs["target"] = np.array(raw[k],np.int32).flatten()[:2]; break
    obs.setdefault("target", np.array([size-2,0],np.int32))
    for k in ("known_map","maze","map","global_map","full_map"):
        if k in raw: obs["known_map"] = np.array(raw[k],np.float32)/6.0; break
    obs.setdefault("known_map", np.zeros((size,size),np.float32))
    for k in ("local","local_view","local_obs","vision"):
        if k in raw: obs["local"] = np.array(raw[k],np.float32)/6.0; break
    if "local" not in obs:
        full = obs["known_map"]; ax2,ay2=int(obs["agent"][0]),int(obs["agent"][1])
        r=3; loc=np.zeros((7,7),np.float32)
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                gx,gy=ax2+dx,ay2+dy
                if 0<=gx<full.shape[1] and 0<=gy<full.shape[0]: loc[dy+r,dx+r]=full[gy,gx]
        obs["local"] = loc
    return obs

def mdist(obs):
    return float(np.abs(obs["agent"].astype(float)-obs["target"].astype(float)).sum())

def encode(raw, src_size):
    obs=unpack(raw,src_size); s=max(src_size-1,1)
    ag=obs["agent"].astype(np.float32)/s
    tg=obs["target"].astype(np.float32)/s
    dl=(obs["target"]-obs["agent"]).astype(np.float32)/s
    d=np.array([float(np.abs(dl).sum())/2.0],np.float32)
    lo=obs["local"].flatten().astype(np.float32)
    km=obs["known_map"].astype(np.float32)
    if km.shape[0]<FINAL_SIZE:
        km=np.pad(km,((0,FINAL_SIZE-src_size),(0,FINAL_SIZE-src_size)),constant_values=0.0)
    return torch.from_numpy(np.concatenate([ag,tg,dl,d,lo,km.flatten()]))

def obs_dim(): return 56 + FINAL_SIZE*FINAL_SIZE
def budget(size): return size*size*2

# ══════════════════════════════════════════════════════════════════
#  NETWORK
# ══════════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    def __init__(self, n_act=5, hidden=512):
        super().__init__()
        in_dim=obs_dim()
        self.shared=nn.Sequential(
            nn.Linear(in_dim,hidden),nn.LayerNorm(hidden),nn.ReLU(),
            nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.ReLU(),
            nn.Linear(hidden,hidden//2),nn.LayerNorm(hidden//2),nn.ReLU(),
        )
        h=hidden//2
        self.V=nn.Sequential(nn.Linear(h,128),nn.ReLU(),nn.Linear(128,1))
        self.A=nn.Sequential(nn.Linear(h,128),nn.ReLU(),nn.Linear(128,n_act))
    def forward(self,x):
        f=self.shared(x); v=self.V(f); a=self.A(f)
        return v+a-a.mean(1,keepdim=True)

def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Not found: {ckpt_path}")
    ckpt=torch.load(ckpt_path,map_location=DEVICE)
    model=DuelingDQN().to(DEVICE)
    model.load_state_dict(ckpt["Q"]); model.eval()
    maze_data = ckpt.get("maze", None)
    print(f"[Checkpoint] t={ckpt.get('t','?')}  in_dim={obs_dim()}")
    print(f"[Checkpoint] maze embedded: {maze_data is not None}\n")
    return model, maze_data

# ══════════════════════════════════════════════════════════════════
#  GREEDY ACTION
# ══════════════════════════════════════════════════════════════════
def is_trapped(raw, size):
    try:
        obs=unpack(raw,size); loc=obs["local"]; r=loc.shape[0]//2
        b=sum(1 for dx,dy in[(1,0),(-1,0),(0,1),(0,-1)]
              if not(0<=r+dy<loc.shape[0] and 0<=r+dx<loc.shape[1])
              or float(loc[r+dy,r+dx]*6)>0.4)
        return b>=4
    except: return False

def act_greedy(model, raw, size):
    s=encode(raw,size).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): q=model(s)[0].clone()
    if not is_trapped(raw,size): q[4]-=0.5
    return int(q.argmax())

# ══════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════
def evaluate(model, env, n_episodes):
    results=[]; max_steps=budget(EVAL_SIZE)
    print(f"[Info] Step budget: {max_steps}\n")
    for ep in range(1, n_episodes+1):
        raw=env_reset(env)
        success=near=False; ep_steps=ep_walls=0
        for _ in range(max_steps):
            action=act_greedy(model,raw,EVAL_SIZE)
            raw,env_r,done,trunc,info=env_step(env,action)
            ep_steps+=1
            if info.get("hit_wall",False): ep_walls+=1
            obs=unpack(raw,EVAL_SIZE)
            if mdist(obs)<=5: near=True
            if done: success=near=True; break
            if trunc: break
        final_d=mdist(unpack(raw,EVAL_SIZE))
        results.append(dict(ep=ep,success=int(success),near_goal=int(near),
                            steps=ep_steps,final_dist=final_d,wall_hits=ep_walls))
        marker="✓ GOAL" if success else ("≈ near" if near else "✗ fail")
        print(f"  ep {ep:4d}/{n_episodes}  {marker}  "
              f"steps={ep_steps:4d}  dist={final_d:5.1f}  walls={ep_walls}")
    return results

def print_summary(results):
    n=len(results)
    succ=[r for r in results if r["success"]]
    fail=[r for r in results if not r["success"]]
    near=[r for r in results if r["near_goal"]]
    sr=len(succ)/n*100; nr=len(near)/n*100
    print(f"\n{'═'*64}")
    print(f"  RESULTS  —  {n} episodes  |  maze {EVAL_SIZE}×{EVAL_SIZE}")
    print(f"{'═'*64}")
    print(f"  ✓  Success rate  : {sr:.1f}%  ({len(succ)}/{n})")
    print(f"  ≈  Near-goal     : {nr:.1f}%  ({len(near)}/{n})")
    print(f"  ─  Avg steps     : {np.mean([r['steps'] for r in results]):.1f}")
    if succ: print(f"  ─  Avg steps (✓) : {np.mean([r['steps'] for r in succ]):.1f}")
    print(f"  ─  Avg final dist: {np.mean([r['final_dist'] for r in results]):.2f}")
    print(f"  ─  Avg wall hits : {np.mean([r['wall_hits'] for r in results]):.1f}")
    print(f"{'═'*64}\n")
    return dict(sr=sr,nr=nr,n_success=len(succ),n_episodes=n,
                avg_steps=np.mean([r['steps'] for r in results]),
                avg_dist=np.mean([r['final_dist'] for r in results]))

def plot_results(results, summary, save_path):
    BG="#f7f9fc"; PAN="#ffffff"; GR="#dde3ed"
    BLUE="#1a6fbd"; RED="#d64045"; GRN="#2e9e5b"; GOLD="#c99a00"; GRAY="#555f6d"
    rc={"axes.facecolor":PAN,"figure.facecolor":BG,"axes.edgecolor":GR,
        "grid.color":GR,"text.color":GRAY,"axes.labelcolor":GRAY,
        "xtick.color":GRAY,"ytick.color":GRAY,"axes.titlecolor":"#1a1f2e",
        "axes.spines.top":False,"axes.spines.right":False}
    eps=[r["ep"] for r in results]
    success=[r["success"] for r in results]
    steps=[r["steps"] for r in results]
    dists=[r["final_dist"] for r in results]
    walls=[r["wall_hits"] for r in results]
    colors=[GRN if s else RED for s in success]
    with plt.rc_context(rc):
        fig=plt.figure(figsize=(20,14),facecolor=BG)
        fig.suptitle(f"DDQN v3 Eval | {EVAL_SIZE}×{EVAL_SIZE} | "
                     f"SR={summary['sr']:.1f}% ({summary['n_success']}/{summary['n_episodes']})",
                     fontsize=16,fontweight="bold",color="#1a1f2e",y=0.98)
        gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.35,
                             left=0.07,right=0.97,top=0.92,bottom=0.07)
        def ax_(r,c):
            ax=fig.add_subplot(gs[r,c]); ax.set_facecolor(PAN)
            ax.grid(True,color=GR,lw=0.7,alpha=0.7)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            return ax
        ax=ax_(0,0)
        ax.bar(eps,[100 if s else 0 for s in success],color=colors,alpha=0.6,width=1.0,linewidth=0)
        if len(eps)>=20:
            ax.plot(np.arange(20,len(eps)+1),
                    np.convolve(success,np.ones(20)/20,mode="valid")*100,
                    color=BLUE,lw=2,label="Rolling SR")
        ax.axhline(summary["sr"],color=GOLD,lw=1.5,ls="--",label=f"Mean={summary['sr']:.1f}%")
        ax.set_ylim(-5,108); ax.set_title("Binary Success",fontweight="bold"); ax.legend(fontsize=9)
        ax=ax_(0,1)
        ax.scatter(eps,dists,c=colors,alpha=0.55,s=20,linewidths=0)
        ax.axhline(5,color=GRN,lw=1.5,ls="--",label="Near-goal (5)")
        ax.axhline(np.mean(dists),color=GOLD,lw=1.5,ls="--",label=f"Mean={np.mean(dists):.1f}")
        ax.set_title("Final Distance",fontweight="bold"); ax.legend(fontsize=9)
        ax=ax_(0,2)
        ax.scatter(eps,steps,c=colors,alpha=0.55,s=20,linewidths=0)
        ax.axhline(np.mean(steps),color=GOLD,lw=1.5,ls="--",label=f"Mean={np.mean(steps):.0f}")
        ax.set_title("Steps per Episode",fontweight="bold"); ax.legend(fontsize=9)
        ax=ax_(1,0)
        succ_d=[r["final_dist"] for r in results if r["success"]]
        fail_d=[r["final_dist"] for r in results if not r["success"]]
        bins=np.linspace(0,max(dists)+1,40)
        if succ_d: ax.hist(succ_d,bins=bins,color=GRN,alpha=0.7,label=f"Success n={len(succ_d)}",edgecolor="none")
        if fail_d: ax.hist(fail_d,bins=bins,color=RED,alpha=0.5,label=f"Failure n={len(fail_d)}",edgecolor="none")
        ax.set_title("Final Distance Distribution",fontweight="bold"); ax.legend(fontsize=9)
        ax=ax_(1,1)
        ax.scatter(eps,walls,c=colors,alpha=0.55,s=20,linewidths=0)
        ax.axhline(np.mean(walls),color=GOLD,lw=1.5,ls="--",label=f"Mean={np.mean(walls):.1f}")
        ax.set_title("Wall Hits",fontweight="bold"); ax.legend(fontsize=9)
        ax=ax_(1,2); ax.axis("off")
        txt=(f"CHECKPOINT\n{os.path.basename(args.checkpoint)}\n\n"
             f"Maze        : {EVAL_SIZE}×{EVAL_SIZE}\n"
             f"Network     : {FINAL_SIZE}×{FINAL_SIZE}\n"
             f"Episodes    : {summary['n_episodes']}\n"
             f"Budget      : {budget(EVAL_SIZE)} steps\n\n"
             f"SUCCESS     : {summary['sr']:.1f}%\n"
             f"Near-goal   : {summary['nr']:.1f}%\n"
             f"Avg steps   : {summary['avg_steps']:.0f}\n"
             f"Avg dist    : {summary['avg_dist']:.2f}\n\n"
             f"Full map revealed\nevery step")
        ax.text(0.05,0.96,txt,transform=ax.transAxes,fontsize=11,
                verticalalignment="top",fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.6",facecolor=PAN,edgecolor=GR,linewidth=1.5))
        plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor=BG)
        plt.close(fig)
        print(f"[Plot saved] → {save_path}")

if __name__ == "__main__":
    model, maze_data = load_model(args.checkpoint)

    # If checkpoint has no maze, try loading from saved pkl file
    if maze_data is None:
        import pickle
        pkl_path = os.path.join(SCRIPT_DIR, "checkpoints_v3", "training_maze.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                maze_data = pickle.load(f)
            print(f"[Maze] Loaded from {pkl_path}")
        else:
            print(f"[Maze] No pkl found at {pkl_path}")
            print(f"       Run save_training_maze.py first!")

    env = make_env(EVAL_SIZE, render=args.render, maze_data=maze_data)
    print(f"[Running {args.episodes} episodes on {EVAL_SIZE}×{EVAL_SIZE}...]\n")
    results  = evaluate(model, env, args.episodes)
    env.close()
    summary  = print_summary(results)
    stem     = os.path.splitext(os.path.basename(args.checkpoint))[0]
    plot_path= os.path.join(os.path.dirname(args.checkpoint),
                            f"eval_{stem}_size{EVAL_SIZE}.png")
    plot_results(results, summary, plot_path)
    print(f"  Done. → {plot_path}\n")