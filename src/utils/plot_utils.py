# FILE: src/utils/plot_utils.py
"""
Simple plotting utility that saves training plots to results/plots.
"""
import os
import matplotlib.pyplot as plt

class TrainingPlotter:
    def __init__(self, out_dir="results/plots"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _moving_average(self, data, window=200):
        if len(data) < window:
            return data
        import numpy as np
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode="valid")

    def plot_all(self, rewards, eval_points, eval_success_rates, eval_avg_steps,
                 epsilons=None, episode_lengths=None):
        # === 1. Raw episode rewards ===
        plt.figure(figsize=(8, 4))
        plt.plot(rewards)
        plt.title("Episode rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "rewards.png"))
        plt.close()

        # === 2. Smoothed rewards ===
        smooth = self._moving_average(rewards, window=max(50, len(rewards)//50 if len(rewards) > 0 else 50))
        plt.figure(figsize=(8, 4))
        plt.plot(smooth)
        plt.title("Smoothed rewards (moving average)")
        plt.xlabel("Episode")
        plt.ylabel("Smoothed reward")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "rewards_smoothed.png"))
        plt.close()

        # === 3. Epsilon decay (NEW progress signal) ===
        if epsilons is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(epsilons)
            plt.title("Epsilon decay during training")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "epsilon_decay.png"))
            plt.close()

        # === 4. Episode length trend (NEW progress signal) ===
        if episode_lengths is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(episode_lengths)
            plt.title("Episode length over time")
            plt.xlabel("Episode")
            plt.ylabel("Steps per episode")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "episode_lengths.png"))
            plt.close()

        if eval_points:
            # === 5. Evaluation success ===
            plt.figure(figsize=(8, 4))
            plt.plot(eval_points, eval_success_rates, marker='o')
            plt.title("Evaluation success rate")
            plt.xlabel("Episode")
            plt.ylabel("Success rate")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "eval_success.png"))
            plt.close()

        # === 2. Smoothed rewards (NEW) ===
        smooth = self._moving_average(rewards, window=max(50, len(rewards)//50 if len(rewards) > 0 else 50))
        plt.figure(figsize=(8, 4))
        plt.plot(smooth)
        plt.title("Smoothed rewards (moving average)")
        plt.xlabel("Episode")
        plt.ylabel("Smoothed reward")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "rewards_smoothed.png"))
        plt.close()

        # === 3. Reward distribution (NEW) ===
        plt.figure(figsize=(6, 4))
        plt.hist(rewards, bins=50)
        plt.title("Reward distribution")
        plt.xlabel("Episode reward")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "reward_histogram.png"))
        plt.close()

        if eval_points:
            # === 4. Evaluation success ===
            plt.figure(figsize=(8, 4))
            plt.plot(eval_points, eval_success_rates, marker='o')
            plt.title("Evaluation success rate")
            plt.xlabel("Episode")
            plt.ylabel("Success rate")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "eval_success.png"))
            plt.close()

            # === 5. Evaluation steps ===
            plt.figure(figsize=(8, 4))
            plt.plot(eval_points, eval_avg_steps, marker='o')
            plt.title("Evaluation average steps")
            plt.xlabel("Episode")
            plt.ylabel("Avg steps")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "eval_steps.png"))
            plt.close()

        if eval_points:
            plt.figure(figsize=(8, 4))
            plt.plot(eval_points, eval_success_rates, marker='o')
            plt.title("Evaluation success rate")
            plt.xlabel("Episode")
            plt.ylabel("Success rate")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "eval_success.png"))
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.plot(eval_points, eval_avg_steps, marker='o')
            plt.title("Evaluation average steps")
            plt.xlabel("Episode")
            plt.ylabel("Avg steps")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "eval_steps.png"))
            plt.close()
