#src/environment/gridworld_maze
import os
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# region GridWorldEnv Class
class GridWorldEnv(gym.Env):
    """
    GridWorld environment with procedurally-generated maze walls, moving key walls and gridlocks,
    teleporting portals (closer/farther), and icons. Supports any square grid size.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
            self,
            render_mode=None,
            size: int = 20,
            maze_type: str = "multiple_paths",  # "single", "multiple_paths"
            n_gridlocks: int | None = None,
            n_key_walls: int | None = None,
            wall_update_freq: int = 3,
            n_portal_pairs: int = 0,
            gridlock_ratio: float | None = None,
            keywall_ratio: float | None = None,
            min_keywall_distance: int | None = None,
    ):
        pygame.init()
        display_info = pygame.display.Info()
        screen_width, screen_height = display_info.current_w, display_info.current_h

        self.size = size
        self.maze_type = maze_type

        default_gridlock_ratio = 0.8
        default_keywall_ratio = 0.8

        gridlock_ratio = float(gridlock_ratio) if gridlock_ratio is not None else default_gridlock_ratio
        keywall_ratio = float(keywall_ratio) if keywall_ratio is not None else default_keywall_ratio

        if n_gridlocks is not None:
            self.n_gridlocks = int(n_gridlocks)
        else:
            self.n_gridlocks = max(2, int(round(self.size * gridlock_ratio)))

        if n_key_walls is not None:
            self.n_key_walls = int(n_key_walls)
        else:
            self.n_key_walls = max(1, int(round(self.size * keywall_ratio)))

        self.k_min_open = max(1, self.n_key_walls // 2)  # at least half always open

        if min_keywall_distance is None:
            self.min_keywall_distance = max(2, self.size // 10)
        else:
            self.min_keywall_distance = int(min_keywall_distance)

        max_window_size = min(screen_width, screen_height) - 100
        self.window_size = max_window_size
        self.cell_size = self.window_size / self.size

        if maze_type == "multiple_paths":
            self.walls = self._generate_maze_walls_multiple_paths(size)
        else:
            self.walls = self._generate_maze_walls(size)

        start_pos = (1, size - 1)
        goal_pos = (size - 2, 0)
        critical_path_cells = self._find_critical_path_cells(start_pos, goal_pos)

        valid_cells = [
            (x, y)
            for y in range(size)
            for x in range(size)
            if (x, y) not in self.walls and (x, y) not in [start_pos, goal_pos]
        ]

        gridlock_candidates = [
            cell for cell in valid_cells
            if cell not in critical_path_cells
        ]
        self.gridlocks = set(random.sample(
            gridlock_candidates,
            min(self.n_gridlocks, len(gridlock_candidates))
        ))

        self.key_walls = {}
        key_wall_candidates = [
            cell for cell in critical_path_cells
            if cell not in self.gridlocks and cell in valid_cells
        ]

        # --------- Even spatial distribution across maze grid zones ----------
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def select_evenly_distributed(candidates, k, size):
            """
            Divide the candidate list into k zones along the start→goal diagonal,
            pick one candidate per zone (farthest from already selected within zone).
            Falls back to farthest-first if a zone is empty.
            Guarantees exactly k picks if enough candidates exist.
            """
            if not candidates or k <= 0:
                return []

            candidates = list(candidates)
            if k == 1:
                return [candidates[len(candidates) // 2]]

            # Sort by diagonal position (x+y), which runs from start (1,size-1) to goal (size-2,0)
            candidates_sorted = sorted(candidates, key=lambda c: c[0] + c[1])
            n = len(candidates_sorted)
            selected = []
            used = set()

            for i in range(k):
                # Compute bucket boundaries
                start_i = (i * n) // k
                end_i   = ((i + 1) * n) // k
                bucket  = [c for c in candidates_sorted[start_i:end_i] if c not in used]

                if not bucket:
                    # Zone empty — pick globally farthest from already selected
                    remaining = [c for c in candidates_sorted if c not in used]
                    if not remaining:
                        break
                    if selected:
                        pick = max(remaining, key=lambda c: min(manhattan(c, s) for s in selected))
                    else:
                        pick = remaining[len(remaining) // 2]
                else:
                    # Pick candidate in zone farthest from already selected
                    if selected:
                        pick = max(bucket, key=lambda c: min(manhattan(c, s) for s in selected))
                    else:
                        pick = bucket[len(bucket) // 2]

                selected.append(pick)
                used.add(pick)

            return selected

        selected = select_evenly_distributed(key_wall_candidates, self.n_key_walls, size)

        # If not enough candidates, reduce n_key_walls to feasible count
        if len(selected) < self.n_key_walls:
            old_n = self.n_key_walls
            self.n_key_walls = len(selected)
            self.k_min_open = max(1, self.n_key_walls // 2)
            print(f"[GridWorldEnv] Warning: requested {old_n} key_walls but only {self.n_key_walls} "
                  f"could be placed. Reduced n_key_walls.")

        # assign to key_walls — high open probability so agent can pass through often
        for pos in selected:
            toggle_prob = random.uniform(0.65, 0.90)   # was 0.25-0.75, now mostly open
            phase = random.randint(0, max(1, self.size // 6))
            self.key_walls[pos] = {"toggle_prob": toggle_prob, "phase": phase}

        self.free = {
            (x, y)
            for y in range(size)
            for x in range(size)
            if (x, y) not in self.walls and (x, y) not in self.gridlocks and (x, y) not in self.key_walls
        }

        # Spaces
        self.vision_radius = 3
        r = self.vision_radius
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
            "target": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
            "local": spaces.Box(low=0, high=6, shape=(2 * r + 1, 2 * r + 1), dtype=np.int8),
            "known_map": spaces.Box(low=0, high=6, shape=(size, size), dtype=np.int8),
        })

        self.action_space = spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Icons — set to None here; loaded lazily in _render_frame
        self.agent_icon = None
        self.target_icon = None
        self.portal_icon_closer = None
        self.portal_icon_farther = None
        self.key_icon = None

        self.n_portal_pairs = max(0, int(n_portal_pairs)) if n_portal_pairs is not None else 0
        self._portals = []

        self.step_count = 0
        self.wall_update_freq = wall_update_freq

        self.prev_distance = None
        self.visited_cells = set()
        self.consecutive_waits = 0
        self.consecutive_wall_hits = 0
        self.last_action = None
        self.steps_since_progress = 0

        self.known_map = None

    # Path analysis & BFS helpers (same as before)
    def _find_critical_path_cells(self, start_pos: tuple[int, int], goal_pos: tuple[int, int]) -> set[tuple[int, int]]:
        critical_cells = set()
        free_cells = {
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if (x, y) not in self.walls
        }

        shortest_path_length = self._bfs_path_length(start_pos, goal_pos, free_cells)

        if shortest_path_length is None:
            return critical_cells

        for cell in free_cells:
            if cell in [start_pos, goal_pos]:
                continue

            test_free_cells = free_cells - {cell}
            test_path_length = self._bfs_path_length(start_pos, goal_pos, test_free_cells)

            if (test_path_length is None or
                    test_path_length > shortest_path_length * 1.3):
                critical_cells.add(cell)

        for pos in [start_pos, goal_pos]:
            for radius in [1, 2]:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) + abs(dy) <= radius:
                            neighbor = (pos[0] + dx, pos[1] + dy)
                            if (0 <= neighbor[0] < self.size and
                                    0 <= neighbor[1] < self.size and
                                    neighbor in free_cells):
                                critical_cells.add(neighbor)

        return critical_cells

    def _bfs_path_length(self, start: tuple[int, int], goal: tuple[int, int],
                         accessible_cells: set[tuple[int, int]]) -> int | None:
        if start not in accessible_cells or goal not in accessible_cells:
            return None

        queue = [(start, 0)]
        visited = {start}

        while queue:
            (x, y), distance = queue.pop(0)

            if (x, y) == goal:
                return distance

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (x + dx, y + dy)
                if (neighbor in accessible_cells and neighbor not in visited):
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return None

    def _bfs_shortest_path(self, start: tuple[int,int], goal: tuple[int,int],
                           accessible_cells: set[tuple[int,int]]) -> list[tuple[int,int]] | None:
        if start not in accessible_cells or goal not in accessible_cells:
            return None
        queue = [(start, [start])]
        visited = {start}
        while queue:
            (x, y), path = queue.pop(0)
            if (x, y) == goal:
                return path
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                n = (x + dx, y + dy)
                if n in accessible_cells and n not in visited:
                    visited.add(n)
                    queue.append((n, path + [n]))
        return None

    # Maze generation (unchanged)
    def _generate_maze_walls(self, size: int) -> set[tuple[int, int]]:
        maze = [[True] * size for _ in range(size)]
        def carve(x: int, y: int):
            dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[ny][nx]:
                    maze[ny][nx] = False
                    maze[y + dy // 2][x + dx // 2] = False
                    carve(nx, ny)
        seed_x, seed_y = 1, size - 3
        maze[seed_y][seed_x] = False
        carve(seed_x, seed_y)
        maze[size - 1][1] = False
        maze[size - 2][1] = False
        maze[0][size - 2] = False
        maze[1][size - 2] = False
        return {(x, y) for y in range(size) for x in range(size) if maze[y][x]}

    def _generate_maze_walls_multiple_paths(self, size: int) -> set[tuple[int, int]]:
        maze = [[True] * size for _ in range(size)]
        def carve(x: int, y: int):
            dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[ny][nx]:
                    maze[ny][nx] = False
                    maze[y + dy // 2][x + dx // 2] = False
                    carve(nx, ny)
        seed_x, seed_y = 1, size - 3
        maze[seed_y][seed_x] = False
        carve(seed_x, seed_y)
        maze[size - 1][1] = False
        maze[size - 2][1] = False
        maze[0][size - 2] = False
        maze[1][size - 2] = False
        wall_positions = []
        for y in range(2, size - 2):
            for x in range(2, size - 2):
                if maze[y][x]:
                    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                    open_neighbors = sum(1 for nx, ny in neighbors
                                         if 0 <= nx < size and 0 <= ny < size and not maze[ny][nx])
                    if open_neighbors >= 2:
                        wall_positions.append((x, y))
        num_additional_paths = max(6, size // 4)
        start = (1, size - 1)
        goal = (size - 2, 0)
        open_cells = {(x, y) for y in range(size) for x in range(size) if not maze[y][x]}
        shortest = self._bfs_shortest_path(start, goal, open_cells)
        def dist_to_path(cell, path):
            if not path:
                return float("inf")
            return min(abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for p in path)
        random.shuffle(wall_positions)
        if shortest:
            wall_positions.sort(key=lambda c: dist_to_path(c, shortest))
        removed_count = 0
        attempts = 0
        max_attempts = max(200, num_additional_paths * 30)
        while removed_count < num_additional_paths and attempts < max_attempts and wall_positions:
            attempts += 1
            pos = wall_positions.pop(0)
            x, y = pos
            maze[y][x] = False
            removed_count += 1
            if random.random() < 0.25:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 2 <= nx < size - 2 and 2 <= ny < size - 2 and maze[ny][nx] and random.random() < 0.35:
                        maze[ny][nx] = False
        if removed_count == 0 and wall_positions:
            for pos in wall_positions[:max(3, num_additional_paths // 2)]:
                maze[pos[1]][pos[0]] = False
        return {(x, y) for y in range(size) for x in range(size) if maze[y][x]}

    # Observation & helpers (unchanged)
    def _get_obs(self):
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        r = self.vision_radius
        local = np.full((2 * r + 1, 2 * r + 1), fill_value=0, dtype=np.int8)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = ax + dx, ay + dy
                if 0 <= x < self.size and 0 <= y < self.size:
                    local[dy + r, dx + r] = self._encode_cell(x, y)
                else:
                    local[dy + r, dx + r] = 0
        return {
            "agent": self._agent_location.copy().astype(np.int32),
            "target": self._target_location.copy().astype(np.int32),
            "local": local,
            "known_map": self.known_map.copy(),
        }

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def _encode_cell(self, x: int, y: int) -> int:
        if (x, y) == tuple(self._target_location):
            return 6
        if (x, y) in self.walls:
            return 2
        if (x, y) in self.gridlocks and (x, y) not in self.free:
            return 3
        if (x, y) in self.key_walls and (x, y) not in self.free:
            return 4
        if any(p["pos"] == (x, y) for p in self._portals):
            return 5
        return 1

    def _init_known_map(self):
        self.known_map = np.zeros((self.size, self.size), dtype=np.int8)
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        self._reveal_from(ax, ay)

    def _reveal_from(self, ax: int, ay: int):
        r = self.vision_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = ax + dx, ay + dy
                if 0 <= x < self.size and 0 <= y < self.size:
                    self.known_map[y, x] = self._encode_cell(x, y)

    def _is_blocked(self, pos: tuple[int, int]) -> bool:
        if pos in self.walls:
            return True
        if pos in self.gridlocks and pos not in self.free:
            return True
        if pos in self.key_walls and pos not in self.free:
            return True
        return False

    def _count_blocked_neighbors(self, pos: tuple[int, int]) -> int:
        x, y = pos
        blocked_count = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (x + dx, y + dy)
            if (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size):
                if self._is_blocked(neighbor):
                    blocked_count += 1
            else:
                blocked_count += 1
        return blocked_count

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = np.array([1, self.size - 1])
        self._target_location = np.array([self.size - 2, 0])

        self._portals = []
        total_portals = 2 * self.n_portal_pairs

        valid_portal_cells = [
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if (x, y) not in self.walls
               and (x, y) not in self.gridlocks
               and (x, y) not in self.key_walls
               and (x, y) != tuple(self._agent_location)
               and (x, y) != tuple(self._target_location)
        ]

        if len(valid_portal_cells) < total_portals:
            valid_portal_cells = [
                (x, y)
                for y in range(self.size)
                for x in range(self.size)
                if (x, y) not in self.walls
                   and (x, y) != tuple(self._agent_location)
                   and (x, y) != tuple(self._target_location)
            ]

        chosen = random.sample(valid_portal_cells, min(total_portals, len(valid_portal_cells)))

        for i, pos in enumerate(chosen):
            ptype = "closer" if i < len(chosen) // 2 else "farther"
            self._portals.append({"pos": pos, "type": ptype})

        self._reset_tracking_vars()
        self._init_known_map()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def _reset_tracking_vars(self):
        self.step_count = 0
        self.consecutive_wall_hits = 0
        self.prev_distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        self.visited_cells = {tuple(self._agent_location)}
        self.consecutive_waits = 0
        self.last_action = None
        self.steps_since_progress = 0

    def _calculate_reward(self, action, info_flags, current_distance):
        if info_flags.get("goal_reached"):
            return 10.0
        reward = -0.01
        if info_flags.get("hit_wall"):
            reward -= 0.05
        if info_flags.get("made_progress"):
            reward += 0.02
        if info_flags.get("used_portal"):
            if info_flags.get("portal_type") == "closer":
                reward += 0.1
            else:
                reward -= 0.1
        return float(reward)

    def step(self, action: int):
        self.step_count += 1
        if self.step_count % self.wall_update_freq == 0:
            self._update_moving_walls()

        direction = self._action_to_direction[action]
        new = np.clip(self._agent_location + direction, 0, self.size - 1)
        pos = (int(new[0]), int(new[1]))

        info_flags = {
            "goal_reached": False,
            "hit_wall": False,
            "gridlock": False,
            "used_portal": False,
            "portal_type": None,
            "made_progress": False,
            "explored_new": False,
            "strategic_wait": False,
        }

        if not self._is_blocked(pos):
            self._agent_location = np.array([pos[0], pos[1]])
            if tuple(self._agent_location) not in self.visited_cells:
                info_flags["explored_new"] = True
                self.visited_cells.add(tuple(self._agent_location))
        else:
            info_flags["hit_wall"] = True

        at_pos = tuple(self._agent_location)
        if at_pos in self.gridlocks and at_pos not in self.free:
            info_flags["gridlock"] = True

        portal_here = next((p for p in self._portals if p["pos"] == at_pos), None)
        if portal_here is not None:
            info_flags["used_portal"] = True
            info_flags["portal_type"] = portal_here["type"]
            if portal_here["type"] == "closer":
                self._teleport_closer()
            else:
                self._teleport_farther()

        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        self._reveal_from(ax, ay)

        done = np.array_equal(self._agent_location, self._target_location)
        if done:
            info_flags["goal_reached"] = True

        current_distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)

        if action == 4:
            self.consecutive_waits += 1
        else:
            self.consecutive_waits = 0

        if self.prev_distance is not None and current_distance < self.prev_distance:
            info_flags["made_progress"] = True
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        reward = self._calculate_reward(action, info_flags, current_distance)

        truncated = False
        if self.step_count >= (self.size * self.size * 2) and not info_flags["goal_reached"]:
            truncated = True
            reward -= 2.0

        self.prev_distance = current_distance
        self.last_action = action

        if self.render_mode == "human":
            self._render_frame()

        full_info = {
            **self._get_info(),
            **info_flags,
            "step_count": self.step_count,
            "consecutive_waits": self.consecutive_waits,
            "consecutive_wall_hits": self.consecutive_wall_hits,
            "steps_since_progress": self.steps_since_progress,
            "current_distance": current_distance,
        }

        done_bool = bool(done)
        return self._get_obs(), float(reward), done_bool, bool(truncated), full_info

    def _teleport_closer(self):
        tx, ty = int(self._target_location[0]), int(self._target_location[1])
        cur_dist = abs(int(self._agent_location[0]) - tx) + abs(int(self._agent_location[1]) - ty)
        candidates = [
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if (x, y) not in self.walls and (x, y) not in self.gridlocks and (x, y) not in self.key_walls
        ]
        closer = [(c, abs(c[0] - tx) + abs(c[1] - ty)) for c in candidates if
                  (abs(c[0] - tx) + abs(c[1] - ty)) < cur_dist]
        if closer:
            new_pos = random.choice([c for c, _ in closer])
            self._agent_location = np.array([new_pos[0], new_pos[1]])
            self.visited_cells.add(new_pos)
        else:
            if (tx, ty) not in self.walls and (tx, ty) not in self.gridlocks and (tx, ty) not in self.key_walls:
                self._agent_location = np.array([tx, ty])

    def _teleport_farther(self):
        tx, ty = int(self._target_location[0]), int(self._target_location[1])
        cur_dist = abs(int(self._agent_location[0]) - tx) + abs(int(self._agent_location[1]) - ty)
        candidates = [
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if (x, y) not in self.walls and (x, y) not in self.gridlocks and (x, y) not in self.key_walls
        ]
        farther = [(c, abs(c[0] - tx) + abs(c[1] - ty)) for c in candidates if
                   (abs(c[0] - tx) + abs(c[1] - ty)) > cur_dist]
        if farther:
            new_pos = random.choice([c for c, _ in farther])
            self._agent_location = np.array([new_pos[0], new_pos[1]])
            self.visited_cells.add(new_pos)

    def _path_exists(self) -> bool:
        start, goal = (1, self.size - 1), (self.size - 2, 0)
        queue, seen = [start], {start}
        while queue:
            x, y = queue.pop(0)
            if (x, y) == goal:
                return True
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if (nx, ny) in self.free and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def _update_moving_walls(self):
        for pos in list(self.gridlocks):
            if random.random() < 0.4:
                self.free.add(pos)
            else:
                self.free.discard(pos)

        opened = []
        for pos in list(self.key_walls.keys()):
            params = self.key_walls[pos]
            toggle_prob = params.get("toggle_prob", 0.4)
            phase = params.get("phase", 0)
            if ((self.step_count + phase) % max(1, self.wall_update_freq)) == 0:
                if random.random() < toggle_prob:
                    self.free.add(pos)
                    opened.append(pos)
                else:
                    self.free.discard(pos)
            else:
                if pos in self.free:
                    opened.append(pos)

        if len(opened) < self.k_min_open:
            closed = [p for p in self.key_walls.keys() if p not in self.free]
            random.shuffle(closed)
            for p in closed[: self.k_min_open - len(opened)]:
                self.free.add(p)

        if not self._path_exists():
            closed_dyn = [p for p in (self.gridlocks | set(self.key_walls.keys())) if p not in self.free]
            if closed_dyn:
                self.free.add(random.choice(closed_dyn))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Triwizard Maze - Multiple Paths")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Load icons if not yet loaded
        if self.agent_icon is None:
            # Try multiple candidate asset paths
            this_dir = os.path.dirname(os.path.abspath(__file__))
            asset_candidates = [
                os.path.join(this_dir, "../../assets"),
                os.path.join(this_dir, "../assets"),
                os.path.join(this_dir, "assets"),
                os.path.join(os.getcwd(), "assets"),
            ]
            asset_path = None
            for candidate in asset_candidates:
                if os.path.exists(os.path.join(candidate, "agent_icon.png")):
                    asset_path = candidate
                    break

            if asset_path is not None:
                try:
                    self.agent_icon  = pygame.image.load(os.path.join(asset_path, "agent_icon.png")).convert_alpha()
                    self.target_icon = pygame.image.load(os.path.join(asset_path, "targets_icon.jpg")).convert()
                    self.key_icon    = pygame.image.load(os.path.join(asset_path, "door.png")).convert_alpha()
                    cs = int(self.cell_size)
                    try:
                        self.portal_icon_closer  = pygame.image.load(os.path.join(asset_path, "teleport_closer.png")).convert_alpha()
                        self.portal_icon_farther = pygame.image.load(os.path.join(asset_path, "teleport_farther.png")).convert_alpha()
                    except Exception:
                        self.portal_icon_closer  = pygame.Surface((cs, cs)); self.portal_icon_closer.fill((0, 0, 255))
                        self.portal_icon_farther = pygame.Surface((cs, cs)); self.portal_icon_farther.fill((255, 255, 0))
                    print(f"[Icons] Loaded from: {asset_path}")
                except Exception as e:
                    print(f"[Icons] Load failed: {e}")
                    asset_path = None

            if asset_path is None:
                print(f"[Icons] Assets not found in any of: {asset_candidates}")
                print(f"[Icons] Falling back to colored squares")
                cs = int(self.cell_size)
                self.agent_icon = pygame.Surface((cs, cs)); self.agent_icon.fill((255, 0, 0))
                self.target_icon = pygame.Surface((cs, cs)); self.target_icon.fill((0, 255, 0))
                self.portal_icon_closer = pygame.Surface((cs, cs)); self.portal_icon_closer.fill((0, 0, 255))
                self.portal_icon_farther = pygame.Surface((cs, cs)); self.portal_icon_farther.fill((255, 255, 0))
                self.key_icon = pygame.Surface((cs, cs)); self.key_icon.fill((128, 0, 128))

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.cell_size
        colors = {"wall": (0, 100, 0), "gridlock": (0, 150, 0), "key": (144, 238, 144)}

        for x, y in self.walls:
            pygame.draw.rect(canvas, colors["wall"], pygame.Rect(x * pix, y * pix, pix, pix))

        for x, y in self.gridlocks:
            col = colors["gridlock"] if (x, y) not in self.free else (255, 255, 255)
            pygame.draw.rect(canvas, col, pygame.Rect(x * pix, y * pix, pix, pix))

        for x, y in self.key_walls.keys():
            if (x, y) not in self.free:
                icon = pygame.transform.scale(self.key_icon, (int(pix), int(pix)))
                canvas.blit(icon, (x * pix, y * pix))

        a_img = pygame.transform.scale(self.agent_icon,          (int(pix), int(pix)))
        t_img = pygame.transform.scale(self.target_icon,         (int(pix), int(pix)))
        p_img_closer  = pygame.transform.scale(self.portal_icon_closer,  (int(pix), int(pix)))
        p_img_farther = pygame.transform.scale(self.portal_icon_farther, (int(pix), int(pix)))

        canvas.blit(t_img, (int(self._target_location[0] * pix), int(self._target_location[1] * pix)))
        canvas.blit(a_img, (int(self._agent_location[0]  * pix), int(self._agent_location[1]  * pix)))

        for p in self._portals:
            if p["type"] == "closer":
                canvas.blit(p_img_closer,  (int(p["pos"][0] * pix), int(p["pos"][1] * pix)))
            else:
                canvas.blit(p_img_farther, (int(p["pos"][0] * pix), int(p["pos"][1] * pix)))

        for i in range(self.size + 1):
            pygame.draw.line(canvas, (180, 180, 180), (0, i * pix), (self.window_size, i * pix))
            pygame.draw.line(canvas, (180, 180, 180), (i * pix, 0), (i * pix, self.window_size))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window:
            pygame.display.quit()
        pygame.quit()
# endregion GridWorldEnv Class