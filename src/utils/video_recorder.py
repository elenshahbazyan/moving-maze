# FILE: src/utils/video_recorder.py
"""
Simple video recorder that saves evaluation episodes as GIFs (if imageio is available).
If imageio is not installed, the recorder degrades gracefully and does not record.
"""
import os
import time

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False


class VideoRecorder:
    def __init__(self, out_dir="results/videos"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def record_episode(self, env, agent, max_steps=1000, filename="episode", greedy=False):
        if not IMAGEIO_AVAILABLE:
            print("[VideoRecorder] imageio not available -- skipping recording (install imageio to enable).")
            # fallback: run a normal greedy episode and return success/steps
            obs, _ = env.reset()
            step_count = 0
            s = agent.state_id(obs["agent"], step_count)
            for t in range(max_steps):
                a = agent.greedy_action(s) if greedy else agent.greedy_action(s)
                next_obs, reward, terminated, truncated, info = env.step(a)
                step_count += 1
                if terminated:
                    return True, step_count
                if truncated:
                    return False, step_count
                s = agent.state_id(next_obs["agent"], step_count)
            return False, step_count

        frames = []
        obs, _ = env.reset()
        step_count = 0
        s = agent.state_id(obs["agent"], step_count)

        for t in range(max_steps):
            a = agent.greedy_action(s) if greedy else agent.greedy_action(s)
            frame = env.render() if hasattr(env, "render") else None
            if frame is not None:
                frames.append(frame.astype('uint8'))
            next_obs, reward, terminated, truncated, info = env.step(a)
            step_count += 1
            if terminated:
                out_path = os.path.join(self.out_dir, f"{filename}.gif")
                try:
                    imageio.mimsave(out_path, frames, fps=6)
                    print(f"Saved video: {out_path}")
                except Exception as e:
                    print(f"Failed to save video: {e}")
                return True, step_count
            if truncated:
                out_path = os.path.join(self.out_dir, f"{filename}_truncated.gif")
                try:
                    imageio.mimsave(out_path, frames, fps=6)
                    print(f"Saved truncated video: {out_path}")
                except Exception as e:
                    print(f"Failed to save video: {e}")
                return False, step_count
            s = agent.state_id(next_obs["agent"], step_count)
        return False, step_count

