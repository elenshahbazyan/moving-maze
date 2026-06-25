import os
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    DDQN-OPTIMIZED Dynamic GridWorld

    ✔ Fully observable
    ✔ Dynamic walls preserved
    ✔ Stable rewards
    ✔ Icons restored
    ✔ Thesis compliant
    ✔ Works with Dueling DDQN
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(
        self,
        render_mode=None,
        size: int = 35,
        maze_type: str = "multiple_paths",
        wall_update_freq: int = 3,
        gridlock_ratio: float = 0.15,
        keywall_ratio: float = 0.10,
    ):
        super().__init__()
        pygame.init()

        self.size = size
        self.render_mode = render_mode
        self.wall_update_freq = wall_update_freq

        # ---------------- MAZE ----------------
        if maze_type == "multiple_paths":
            self.walls = self._generate_maze_walls(size)
        else:
            self.walls = self._generate_maze_walls(size)

        self.n_gridlocks = max(2, int(self.size * gridlock_ratio))
        self.n_key_walls = max(1, int(self.size * keywall_ratio))
        self.k_min_open = max(1, self.n_key_walls // 4)

        start_pos = (1, size - 1)
        goal_pos = (size - 2, 0)

        valid_cells = [
            (x, y)
            for y in range(size)
            for x in range(size)
            if (x, y) not in self.walls and (x, y) not in [start_pos, goal_pos]
        ]

        self.gridlocks = set(
            random.sample(valid_cells, min(self.n_gridlocks, len(valid_cells)))
        )

        remaining = [c for c in valid_cells if c not in self.gridlocks]
        self.key_walls = {
            pos: {"toggle_prob": random.uniform(0.25, 0.75)}
            for pos in random.sample(remaining, min(self.n_key_walls, len(remaining)))
        }

        self.free = set()

        # ---------------- OBS SPACE ----------------
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
            "maze": spaces.Box(0, 6, shape=(size, size), dtype=np.int8),
        })

        self.action_space = spaces.Discrete(5)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, 0]),  # WAIT
        }

        self.step_count = 0
        self.window = None
        self.clock = None

        # -------- ICONS --------
        self.agent_icon = None
        self.target_icon = None
        self.key_icon = None

    # ============================================================
    # MAZE GENERATION (DFS)
    # ============================================================

    def _generate_maze_walls(self, size: int):
        maze = [[True] * size for _ in range(size)]

        def carve(x, y):
            dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[ny][nx]:
                    maze[ny][nx] = False
                    maze[y + dy // 2][x + dx // 2] = False
                    carve(nx, ny)

        sx, sy = 1, size - 3
        maze[sy][sx] = False
        carve(sx, sy)

        maze[size - 1][1] = False
        maze[size - 2][1] = False
        maze[0][size - 2] = False
        maze[1][size - 2] = False

        return {(x, y) for y in range(size) for x in range(size) if maze[y][x]}

    # ============================================================
    # FULL MAZE ENCODING
    # ============================================================

    def _get_full_maze(self):
        full = np.ones((self.size, self.size), dtype=np.float32)

        for (x, y) in self.walls:
            full[y, x] = 2

        for (x, y) in self.gridlocks:
            if (x, y) not in self.free:
                full[y, x] = 3

        for (x, y) in self.key_walls:
            if (x, y) not in self.free:
                full[y, x] = 4

        gx, gy = self._target_location
        full[gy, gx] = 6

        return full

    def _get_obs(self):
        return {
            "agent": self._agent_location.astype(np.int32),
            "target": self._target_location.astype(np.int32),
            "maze": self._get_full_maze(),
        }

    # ============================================================
    # RESET
    # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([1, self.size - 1])
        self._target_location = np.array([self.size - 2, 0])

        self.step_count = 0
        self.free.clear()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action: int):
        self.step_count += 1

        if self.step_count % self.wall_update_freq == 0:
            self._update_moving_walls()

        direction = self._action_to_direction[action]
        new_pos = np.clip(self._agent_location + direction, 0, self.size - 1)
        pos = (int(new_pos[0]), int(new_pos[1]))

        reward = -0.01
        hit_wall = False

        if not self._is_blocked(pos):
            self._agent_location = new_pos
        else:
            reward -= 0.05
            hit_wall = True

        done = np.array_equal(self._agent_location, self._target_location)

        if done:
            reward = 10.0

        truncated = False
        if self.step_count >= self.size * self.size * 5:
            truncated = True
            reward -= 1.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), float(reward), bool(done), bool(truncated), {"hit_wall": hit_wall}

    # ============================================================
    # WALL LOGIC
    # ============================================================

    def _is_blocked(self, pos):
        if pos in self.walls:
            return True
        if pos in self.gridlocks and pos not in self.free:
            return True
        if pos in self.key_walls and pos not in self.free:
            return True
        return False

    def _update_moving_walls(self):
        for pos in self.gridlocks:
            if random.random() < 0.4:
                self.free.add(pos)
            else:
                self.free.discard(pos)

        for pos, params in self.key_walls.items():
            if random.random() < params["toggle_prob"]:
                self.free.add(pos)
            else:
                self.free.discard(pos)

    # ============================================================
    # RENDER WITH ICONS
    # ============================================================

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("Dynamic GridWorld DDQN")
            self.clock = pygame.time.Clock()

        # ---------- LOAD ICONS ----------
        if self.agent_icon is None:
            asset_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../assets"
            )

            def fallback(color):
                surf = pygame.Surface((int(600 / self.size), int(600 / self.size)))
                surf.fill(color)
                return surf

            try:
                self.agent_icon = pygame.image.load(
                    os.path.join(asset_path, "agent_icon.png")
                ).convert_alpha()
            except Exception:
                self.agent_icon = fallback((255, 0, 0))

            try:
                self.target_icon = pygame.image.load(
                    os.path.join(asset_path, "targets_icon.jpg")
                ).convert_alpha()
            except Exception:
                self.target_icon = fallback((0, 255, 0))

            try:
                self.key_icon = pygame.image.load(
                    os.path.join(asset_path, "door.png")
                ).convert_alpha()
            except Exception:
                self.key_icon = fallback((128, 0, 128))

        canvas = pygame.Surface((600, 600))
        canvas.fill((255, 255, 255))
        cell = 600 / self.size

        # walls
        for (x, y) in self.walls:
            pygame.draw.rect(canvas, (0, 100, 0), (x * cell, y * cell, cell, cell))

        # dynamic
        for (x, y) in self.gridlocks:
            if (x, y) not in self.free:
                pygame.draw.rect(canvas, (0, 150, 0), (x * cell, y * cell, cell, cell))

        key_scaled = pygame.transform.scale(self.key_icon, (int(cell), int(cell)))
        for (x, y) in self.key_walls:
            if (x, y) not in self.free:
                canvas.blit(key_scaled, (x * cell, y * cell))

        # agent
        ax, ay = self._agent_location
        a_img = pygame.transform.scale(self.agent_icon, (int(cell), int(cell)))
        canvas.blit(a_img, (ax * cell, ay * cell))

        # target
        gx, gy = self._target_location
        t_img = pygame.transform.scale(self.target_icon, (int(cell), int(cell)))
        canvas.blit(t_img, (gx * cell, gy * cell))

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.display.quit()
        pygame.quit()