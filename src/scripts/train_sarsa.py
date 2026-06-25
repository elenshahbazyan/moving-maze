

import os
import argparse
import csv
import pickle
import numpy as np
from time import time

try:
    from src.environment.gridworld_maze import GridWorldEnv
    from src.agent.sarsa import SARSAAgent
    from src.utils.video_recorder import VideoRecorder
    from src.utils.plot_utils import TrainingPlotter
except Exception:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.environment.gridworld_maze import GridWorldEnv
    from src.agent.sarsa import SARSAAgent
    from src.utils.video_recorder import VideoRecorder
    from src.utils.plot_utils import TrainingPlotter


def save_maze_config(env, filepath="results/checkpoints/maze_config.pkl"):
    config = {
        'size': env.size,
        'maze_type': env.maze_type,
        'walls': env.walls,
        'gridlocks': env.gridlocks,
        'key_walls': env.key_walls,
        'free': env.free,
        'n_gridlocks': env.n_gridlocks,
        'n_key_walls': env.n_key_walls,
        'n_portal_pairs': env.n_portal_pairs,
        'wall_update_freq': env.wall_update_freq,
        'min_keywall_distance': env.min_keywall_distance,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(config, f)
    print(f"Saved maze configuration to: {filepath}")
    return config


def train(env, agent: SARSAAgent, episodes: int = 30000, max_steps: int | None = None,
          eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_frac: float = 0.6,
          eval_every: int = 1000, eval_episodes: int = 20, record_video: bool = False,
          video_every: int = 5000):

    if max_steps is None:
        max_steps = env.size * env.size * 2

    eps_decay = max(1, int(episodes * eps_decay_frac))
    best_eval = -1.0

    # Logging containers
    rewards = []
    epsilons = []
    episode_lengths = []  # NEW
    eval_points = []
    eval_success_rates = []
    eval_avg_steps = []

    os.makedirs(os.path.abspath(os.path.join("results", "checkpoints")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join("results", "logs")), exist_ok=True)

    video_recorder = VideoRecorder() if record_video else None
    plotter = TrainingPlotter()

    print(f"Starting training for {episodes} episodes...")
    print(f"Environment size: {env.size}x{env.size}")
    print(f"Agent state space: {agent.n_states} states, {agent.n_actions} actions")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        step_count = 0
        s = agent.state_id(obs["agent"], step_count)
        eps = max(eps_end, eps_start * (1 - (ep / eps_decay))) if ep <= eps_decay else eps_end
        a = agent.act_epsilon(s, eps)

        total_r = 0.0
        steps_this_ep = 0
        for t in range(max_steps):
            next_obs, reward, terminated, truncated, info = env.step(a)
            step_count += 1
            steps_this_ep += 1
            done = bool(terminated or truncated)
            s2 = agent.state_id(next_obs["agent"], step_count)
            a2 = agent.act_epsilon(s2, eps)

            agent.update(s, a, reward, s2, a2)

            s, a = s2, a2
            total_r += reward
            if done:
                break

        rewards.append(total_r)
        epsilons.append(eps)
        episode_lengths.append(steps_this_ep)  # NEW

        # Evaluation
        if ep % eval_every == 0 or ep == 1:
            should_record = record_video and (ep % video_every == 0 or ep == 1)

            succ_rate, avg_steps = evaluate(
                env, agent, episodes=eval_episodes,
                max_steps=max_steps, video_recorder=video_recorder if should_record else None,
                episode_num=ep
            )

            eval_points.append(ep)
            eval_success_rates.append(succ_rate)
            eval_avg_steps.append(avg_steps)

            video_msg = " [VIDEO RECORDED]" if should_record else ""
            print(f"Ep {ep}/{episodes}  reward={total_r:.3f}  eps={eps:.3f}  "
                  f"eval_success={succ_rate:.2%}  avg_steps={avg_steps:.1f}{video_msg}")

            if succ_rate > best_eval:
                best_eval = succ_rate
                agent.save(os.path.abspath(os.path.join("results", "checkpoints", "best_Q.npy")))

        else:
            if ep % (max(1, episodes // 20)) == 0:
                print(f"Ep {ep}/{episodes}  reward={total_r:.3f}  eps={eps:.3f}")

    csv_path = (os.path.abspath(os.path.join("results", "logs", "train_log.csv")))
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "reward", "epsilon", "episode_length"])  # UPDATED
        for i, (r, e, l) in enumerate(zip(rewards, epsilons, episode_lengths), start=1):
            writer.writerow([i, r, e, l])
    print(f"Saved training log to: {csv_path}")

    # UPDATED: pass more metrics to plotter
    plotter.plot_all(rewards, eval_points, eval_success_rates, eval_avg_steps,
                     epsilons=epsilons, episode_lengths=episode_lengths)

    print(f"Training complete!")
    print(f"Best evaluation success rate: {best_eval:.2%}")

    return agent


def evaluate(env, agent: SARSAAgent, episodes: int = 20, max_steps: int | None = None,
             render: bool = False, video_recorder: VideoRecorder = None, episode_num: int = 0):
    if max_steps is None:
        max_steps = env.size * env.size * 2

    successes = 0
    steps_list = []
    video_recorded = False

    video_env = None
    if video_recorder is not None:
        try:
            video_env = GridWorldEnv(
                render_mode="rgb_array",
                size=env.size,
                maze_type=getattr(env, 'maze_type', 'multiple_paths'),
                n_gridlocks=getattr(env, 'n_gridlocks', 10),
                n_key_walls=getattr(env, 'n_key_walls', 8),
                n_portal_pairs=getattr(env, 'n_portal_pairs', 0)
            )
            video_env.walls = env.walls.copy()
            video_env.key_walls = {k: v.copy() for k, v in env.key_walls.items()}
            video_env.gridlocks = env.gridlocks.copy()
            video_env.free = env.free.copy()
        except Exception as e:
            print(f"Warning: Could not create video environment: {e}")
            video_recorder = None

    for ep in range(episodes):
        current_env = video_env if (video_recorder and not video_recorded and ep == 0) else env
        recording_this_ep = (current_env is video_env and video_recorder and not video_recorded)

        if recording_this_ep:
            success, total_steps = video_recorder.record_episode(
                current_env, agent, max_steps=max_steps,
                filename=f"eval_ep{episode_num:06d}", greedy=True
            )
            if success:
                successes += 1
                video_recorded = True
            steps_list.append(total_steps)
        else:
            obs, _ = current_env.reset()
            step_count = 0
            s = agent.state_id(obs["agent"], step_count)
            total_steps = 0

            for t in range(max_steps):
                a = agent.greedy_action(s)
                next_obs, reward, terminated, truncated, info = current_env.step(a)
                step_count += 1
                total_steps += 1

                if render and hasattr(current_env, "render"):
                    current_env.render()

                if terminated:
                    successes += 1
                    break

                if truncated:
                    break

                s = agent.state_id(next_obs["agent"], step_count)

            steps_list.append(total_steps)

    if video_env is not None:
        try:
            video_env.close()
        except:
            pass

    succ_rate = successes / episodes
    avg_steps = float(np.mean(steps_list))
    return succ_rate, avg_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SARSA agent on GridWorld maze")
    parser.add_argument("--episodes", type=int, default=30000, help="Number of training episodes")
    parser.add_argument("--size", type=int, default=40, help="Grid size")
    parser.add_argument("--n_gridlocks", type=int, default=40, help="Number of gridlocks")
    parser.add_argument("--n_key_walls", type=int, default=12, help="Number of key walls")
    parser.add_argument("--n_portal_pairs", type=int, default=0, help="Number of portal pairs")
    parser.add_argument("--include_time_mod", action="store_true", help="Include time in state")
    parser.add_argument("--render_eval", action="store_true", help="Render during final evaluation")
    parser.add_argument("--record_video", action="store_true", help="Record evaluation videos")
    parser.add_argument("--video_every", type=int, default=5000, help="Record video every N episodes")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate every N episodes")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--alpha", type=float, default=0.25, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--eps_decay_frac", type=float, default=0.6, help="Fraction of episodes to decay epsilon")
    args = parser.parse_args()

    print("=" * 60)
    print("SARSA AGENT TRAINING")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Grid size: {args.size}x{args.size}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Gridlocks: {args.n_gridlocks}")
    print(f"  Key walls: {args.n_key_walls}")
    print(f"  Portal pairs: {args.n_portal_pairs}")
    print(f"  Learning rate (alpha): {args.alpha}")
    print(f"  Discount factor (gamma): {args.gamma}")
    print(f"  Epsilon: {args.eps_start} -> {args.eps_end}")
    print(f"  Time-based state: {args.include_time_mod}")
    print("=" * 60 + "\n")

    env = GridWorldEnv(
        render_mode=None,
        size=args.size,
        maze_type="multiple_paths",
        n_gridlocks=args.n_gridlocks,
        n_key_walls=args.n_key_walls,
        n_portal_pairs=args.n_portal_pairs
    )

    save_maze_config(env)

    time_mod = env.wall_update_freq if args.include_time_mod else 1
    agent = SARSAAgent(
        size=args.size,
        n_actions=env.action_space.n,
        time_mod=time_mod,
        alpha=args.alpha,
        gamma=args.gamma
    )

    agent = train(
        env, agent,
        episodes=args.episodes,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_frac=args.eps_decay_frac,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        record_video=args.record_video,
        video_every=args.video_every
    )

    if args.render_eval:
        print("\nRunning final evaluation with rendering...")
        succ, avg = evaluate(env, agent, episodes=5, render=True)
        print(f"Final evaluation -> success: {succ:.2%}, avg_steps: {avg:.1f}")

    agent.save("results/checkpoints/final_Q.npy")
    print("\nSaved final model to: results/checkpoints/final_Q.npy")

    env.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Files saved:")
    print("  - results/checkpoints/best_Q.npy (best performing model)")
    print("  - results/checkpoints/final_Q.npy (final model)")
    print("  - results/checkpoints/maze_config.pkl (maze configuration)")
    print("  - results/logs/train_log.csv (training logs)")
    print("  - results/plots/*.png (training plots)")
    print("\nTo evaluate on the same maze, run:")
    print("  python src/scripts/train_sarsa.py --size 40 --episodes 30000 --record_video --render_eval")
    print("=" * 60)
