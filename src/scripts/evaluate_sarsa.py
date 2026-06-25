# src/scripts/evaluate_sarsa.py
import os
import argparse
import pickle
import numpy as np
from time import time
import pygame

try:
    from src.environment.gridworld_maze import GridWorldEnv
    from src.agent.sarsa import SARSAAgent
    from src.utils.video_recorder import VideoRecorder
except Exception:
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.environment.gridworld_maze import GridWorldEnv
    from src.agent.sarsa import SARSAAgent
    from src.utils.video_recorder import VideoRecorder


def save_maze_config(env, filepath="results/checkpoints/maze_config.pkl"):
    """Save maze configuration to file."""
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


def load_maze_config(filepath="results/checkpoints/maze_config.pkl"):
    """Load maze configuration from file."""
    with open(filepath, 'rb') as f:
        config = pickle.load(f)
    print(f"Loaded maze configuration from: {filepath}")
    return config


def create_env_from_config(config, render_mode=None):
    """Create environment with saved maze configuration."""
    env = GridWorldEnv(
        render_mode=render_mode,
        size=config['size'],
        maze_type=config['maze_type'],
        n_gridlocks=config['n_gridlocks'],
        n_key_walls=config['n_key_walls'],
        n_portal_pairs=config['n_portal_pairs'],
        wall_update_freq=config.get('wall_update_freq', 3),
        min_keywall_distance=config.get('min_keywall_distance', None),
    )

    # Override the randomly generated maze with saved configuration
    env.walls = config['walls'].copy()
    env.gridlocks = config['gridlocks'].copy()
    env.key_walls = {k: v.copy() for k, v in config['key_walls'].items()}
    env.free = config['free'].copy()

    return env


def evaluate_agent(env, agent, episodes=50, max_steps=600, render=False,
                   video_recorder=None, verbose=True, render_speed=10):
    """
    Evaluate agent on the environment.

    Args:
        env: GridWorld environment
        agent: Trained SARSA agent
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        video_recorder: Optional VideoRecorder instance
        verbose: Print detailed statistics
        render_speed: FPS for rendering (higher = faster)

    Returns:
        dict: Evaluation metrics
    """
    successes = 0
    steps_list = []
    rewards_list = []

    # Set render speed if rendering
    if render and hasattr(env, 'metadata'):
        original_fps = env.metadata.get("render_fps", 4)
        env.metadata["render_fps"] = render_speed

    for ep in range(episodes):
        obs, _ = env.reset()
        step_count = 0
        s = agent.state_id(obs["agent"], step_count)

        total_reward = 0.0
        total_steps = 0

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Episode {ep + 1}/{episodes} - Starting...")
            print(f"{'=' * 60}")

        for t in range(max_steps):
            # Use greedy action (no exploration)
            a = agent.greedy_action(s)
            next_obs, reward, terminated, truncated, info = env.step(a)

            step_count += 1
            total_steps += 1
            total_reward += reward

            if render:
                env.render()
                # Handle pygame events to prevent window from freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nRendering interrupted by user")
                        env.close()
                        return None

            if verbose and render:
                # Print step info during rendering
                action_names = {0: "RIGHT", 1: "UP", 2: "LEFT", 3: "DOWN", 4: "WAIT"}
                print(f"Step {total_steps}: Action={action_names.get(a, a)} | "
                      f"Pos={tuple(next_obs['agent'])} | Reward={reward:.3f}")

            if terminated:
                successes += 1
                if verbose:
                    print(f"\n🎉 SUCCESS! Reached goal in {total_steps} steps!")
                break

            if truncated:
                if verbose:
                    print(f"\n⏱️ TRUNCATED after {total_steps} steps")
                break

            s = agent.state_id(next_obs["agent"], step_count)

        steps_list.append(total_steps)
        rewards_list.append(total_reward)

        if verbose:
            status = "✅ SUCCESS" if terminated else "❌ FAILED"
            print(f"\n{status} | Steps: {total_steps} | Total Reward: {total_reward:.3f}")

            if render and ep < episodes - 1:
                print(f"\nPress SPACE to continue to next episode...")
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nEvaluation stopped by user")
                            env.close()
                            return None
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                print("\nEvaluation stopped by user")
                                env.close()
                                return None

    # Restore original FPS
    if render and hasattr(env, 'metadata'):
        env.metadata["render_fps"] = original_fps

    # Calculate metrics
    success_rate = successes / episodes
    avg_steps = float(np.mean(steps_list))
    std_steps = float(np.std(steps_list))
    avg_reward = float(np.mean(rewards_list))
    std_reward = float(np.std(rewards_list))

    metrics = {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'std_steps': std_steps,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'successes': successes,
        'total_episodes': episodes,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Success Rate:     {success_rate:.2%} ({successes}/{episodes})")
        print(f"Average Steps:    {avg_steps:.2f} ± {std_steps:.2f}")
        print(f"Average Reward:   {avg_reward:.3f} ± {std_reward:.3f}")
        print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SARSA agent on saved maze")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/best_Q.npy",
                        help="Path to saved Q-table")
    parser.add_argument("--maze_config", type=str, default="results/checkpoints/maze_config.pkl",
                        help="Path to saved maze configuration")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=600,
                        help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--render_speed", type=int, default=10,
                        help="Rendering speed (FPS, higher = faster)")
    parser.add_argument("--record_video", action="store_true",
                        help="Record video of evaluation")
    parser.add_argument("--include_time_mod", action="store_true",
                        help="Include time in state (must match training)")
    parser.add_argument("--no_step_info", action="store_true",
                        help="Don't print step-by-step information during rendering")

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    if not os.path.exists(args.maze_config):
        print(f"Error: Maze configuration not found at {args.maze_config}")
        print("\nTo save maze configuration during training, modify train_sarsa.py:")
        print("  from evaluate_sarsa import save_maze_config")
        print("  save_maze_config(env)  # Add this after creating env")
        return

    # Load maze configuration
    maze_config = load_maze_config(args.maze_config)

    # Create environment with same maze
    render_mode = "human" if args.render else None
    env = create_env_from_config(maze_config, render_mode=render_mode)

    print(f"\n{'=' * 60}")
    print("ENVIRONMENT LOADED")
    print(f"{'=' * 60}")
    print(f"  Size: {env.size}x{env.size}")
    print(f"  Walls: {len(env.walls)}")
    print(f"  Gridlocks: {len(env.gridlocks)}")
    print(f"  Key Walls: {len(env.key_walls)}")
    print(f"  Portal Pairs: {env.n_portal_pairs}")
    print(f"{'=' * 60}\n")

    # Create agent
    time_mod = env.wall_update_freq if args.include_time_mod else 1
    agent = SARSAAgent(
        size=env.size,
        n_actions=env.action_space.n,
        time_mod=time_mod,
        alpha=0.4,
        gamma=0.99
    )

    # Load trained Q-table
    agent.load(args.checkpoint)
    print(f"Loaded Q-table from: {args.checkpoint}")

    # Initialize video recorder if requested
    video_recorder = None
    if args.record_video:
        video_recorder = VideoRecorder()
        print("Video recording enabled")

    # Evaluate
    print(f"\nEvaluating agent for {args.episodes} episodes...")
    if args.render:
        print(f"Rendering at {args.render_speed} FPS")
        print("\nControls:")
        print("  - SPACE: Continue to next episode")
        print("  - ESC: Stop evaluation")
        print("  - Close window: Stop evaluation")
    print("-" * 60)

    start_time = time()
    metrics = evaluate_agent(
        env, agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        video_recorder=video_recorder,
        verbose=not args.no_step_info,
        render_speed=args.render_speed
    )

    if metrics is None:
        print("\nEvaluation was interrupted")
        env.close()
        return

    eval_time = time() - start_time

    print(f"\nEvaluation completed in {eval_time:.2f} seconds")

    # Record a single video episode if requested (without rendering window)
    if args.record_video:
        print("\nRecording video of one episode...")
        video_env = create_env_from_config(maze_config, render_mode="rgb_array")
        success, steps = video_recorder.record_episode(
            video_env, agent,
            max_steps=args.max_steps,
            filename="evaluation_demo",
            greedy=True
        )
        video_env.close()
        print(f"Video recorded: {'SUCCESS' if success else 'FAILED'} in {steps} steps")

    env.close()


if __name__ == "__main__":
    main()