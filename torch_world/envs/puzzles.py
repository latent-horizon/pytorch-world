"""Exact, inspectable planning puzzles with pixels and irreversible state changes."""

import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition

# # wall, A agent, G goal, K key, D door, B box, 1/2 ordered checkpoints.
LEVELS = {
    "maze": ("#########", "#A..#..G#", "#.#.#.#.#", "#.#...#.#", "#.#####.#", "#.......#", "#########"),
    "key_door": ("#########", "#A..#..G#", "#...D...#", "#...#...#", "#.K.#...#", "#...#...#", "#########"),
    "sokoban": ("#########", "#.......#", "#..A....#", "#..B.G..#", "#.......#", "#.......#", "#########"),
    "ice": ("#########", "#A.....##", "#..#....#", "#.......#", "#....#..#", "#......G#", "#########"),
    "checkpoints": ("#########", "#A..#..G#", "#...#...#", "#.......#", "#.1.#.2.#", "#...#...#", "#########"),
}


class PlanningPuzzle(BenchmarkEnv):
    """Grid navigation, key/door, box pushing, ice momentum or ordered visits.

    Two normalized action components choose the dominant cardinal direction;
    magnitudes below 0.5 wait. Coordinates are (x,y). State is
    (agent_x, agent_y, has_key, checkpoint_count, box_x, box_y).
    ``level`` accepts an ASCII map to define a new puzzle without new code.
    """

    action_dim, state_dim, radius = 2, 6, 0.0

    def __init__(
        self,
        task="maze",
        *,
        level=None,
        batch_size=1,
        size=None,
        device=None,
        max_episode_steps=100,
        quality="high",
        render_size=None,
    ):
        super().__init__(batch_size, size, device, max_episode_steps, quality=quality, render_size=render_size)
        if task not in LEVELS:
            raise ValueError(f"Unknown puzzle {task!r}; choose from {tuple(LEVELS)}")
        self._sprite_cache = {}
        self.task = task
        self.level = tuple(level or LEVELS[task])
        if len(self.level) < 3 or len(set(map(len, self.level))) != 1:
            raise ValueError("level must be a rectangular map with at least three rows")
        self.height, self.width = len(self.level), len(self.level[0])
        cells = {}
        for y, line in enumerate(self.level):
            for x, char in enumerate(line):
                if char not in "#. AGKDB12":
                    raise ValueError(f"Unknown map symbol {char!r}")
                cells.setdefault(char, []).append((x, y))
        for char in "AG" + (
            "KD" if task == "key_door" else "B" if task == "sokoban" else "12" if task == "checkpoints" else ""
        ):
            if len(cells.get(char, [])) != 1:
                raise ValueError(f"level must contain exactly one {char!r}")
        self.walls = set(cells.get("#", []))
        self._wall_mask = torch.zeros(self.height, self.width, dtype=torch.bool, device=self.device)
        wall_xy = torch.tensor(sorted(self.walls), device=self.device, dtype=torch.long).reshape(-1, 2)
        self._wall_mask[wall_xy[:, 1], wall_xy[:, 0]] = True
        self.target = cells["G"][0]
        self.key = cells.get("K", [None])[0]
        self.door = cells.get("D", [None])[0]
        self.checkpoints = [cells[c][0] for c in "12" if c in cells]
        self.initial = (*cells["A"][0], 0, 0, *cells.get("B", [(-1, -1)])[0])
        self.state = torch.tensor(self.initial, dtype=torch.float32, device=self.device).repeat(batch_size, 1)
        self.goal = self.state.clone()
        self.goal[:, :2] = torch.tensor(self.target, device=self.device)
        if task == "key_door":
            self.goal[:, 2] = 1
        if task == "checkpoints":
            self.goal[:, 3] = len(self.checkpoints)
        if task == "sokoban":
            self.goal[:, 4:6] = torch.tensor(self.target, device=self.device)
            x, y = self.target
            neighbors = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            standing = next((xy for xy in neighbors if self._free(xy, False)), None)
            if standing is None:
                raise ValueError("Sokoban goal needs an adjacent free cell for a valid GT configuration")
            self.goal[:, :2] = torch.tensor(standing, device=self.device)

    def _free(self, xy, key):
        x, y = xy
        return 0 <= x < self.width and 0 <= y < self.height and xy not in self.walls and (xy != self.door or key)

    def advance(self, state, direction):
        """Pure integer dynamics, useful for contributor level validation/search."""
        x, y, key, progress, bx, by = map(int, state)
        dx, dy = direction
        for _ in range(max(self.width, self.height) if self.task == "ice" else 1):
            nx, ny = x + dx, y + dy
            if not self._free((nx, ny), key):
                break
            if self.task == "sokoban" and (nx, ny) == (bx, by):
                if not self._free((bx + dx, by + dy), key):
                    break
                bx, by = bx + dx, by + dy
            x, y = nx, ny
            key = int(bool(key) or (x, y) == self.key)
            if progress < len(self.checkpoints) and (x, y) == self.checkpoints[progress]:
                progress += 1
            if (x, y) == self.target:
                break
        return x, y, key, progress, bx, by

    def reference_policy(self):
        from torch_world.envs.references import ActionSequence, puzzle_actions

        return ActionSequence(self, puzzle_actions(self))

    def reset(self, state=None, generator=None, *, seed=None):
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        # These are fixed benchmark levels; seeds do not change the problem.
        self._reset_episode()
        state = torch.tensor(self.initial, device=self.device).repeat(self.batch_size, 1) if state is None else state
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.shape != (self.batch_size, self.state_dim) or not torch.isfinite(state).all():
            raise ValueError("Invalid puzzle state shape or values")
        if not torch.equal(state, state.round()):
            raise ValueError("Puzzle state coordinates and counters must be integers")
        for row in state.tolist():
            if (
                not self._free(tuple(row[:2]), row[2])
                or row[2] not in (0, 1)
                or not 0 <= row[3] <= len(self.checkpoints)
            ):
                raise ValueError("Invalid puzzle position, key or checkpoint state")
            if self.task == "sokoban" and (not self._free(tuple(row[4:6]), row[2]) or row[:2] == row[4:6]):
                raise ValueError("Invalid box position")
        self.state = state.clone()
        return self.render()

    def _free_batch(self, xy, key):
        x, y = xy.long().unbind(-1)
        inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        free = inside & ~self._wall_mask[y.clamp(0, self.height - 1), x.clamp(0, self.width - 1)]
        if self.door is not None:
            door = torch.tensor(self.door, device=self.device)
            free &= (xy != door).any(-1) | key.bool()
        return free

    def _advance_batch(self, state, direction, moving):
        """Vectorized lanes; only ice's sequential slide requires repeated updates."""
        state = state.clone()
        for _ in range(max(self.width, self.height) if self.task == "ice" else 1):
            next_xy = state[:, :2] + direction
            moving = moving & self._free_batch(next_xy, state[:, 2])
            if self.task == "sokoban":
                pushing = (next_xy == state[:, 4:6]).all(-1)
                next_box = state[:, 4:6] + direction
                moving &= ~pushing | self._free_batch(next_box, state[:, 2])
                state[:, 4:6] = torch.where((moving & pushing)[:, None], next_box, state[:, 4:6])
            state[:, :2] = torch.where(moving[:, None], next_xy, state[:, :2])
            if self.key is not None:
                acquired = (state[:, :2] == torch.tensor(self.key, device=self.device)).all(-1)
                state[:, 2] = torch.maximum(state[:, 2], acquired.float())
            for j, checkpoint in enumerate(self.checkpoints):
                reached = (state[:, :2] == torch.tensor(checkpoint, device=self.device)).all(-1)
                state[:, 3] += reached & (state[:, 3] == j)
            moving &= (state[:, :2] != torch.tensor(self.target, device=self.device)).any(-1)
        return state

    @staticmethod
    def _directions(action):
        """Decode normalized controls identically for simulation and offline pairs."""
        axis = action.abs().argmax(-1, keepdim=True)
        direction = torch.zeros_like(action).scatter_(1, axis, action.gather(1, axis).sign())
        return direction, action.abs().amax(-1) >= 0.5

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        direction, moving = self._directions(action)
        self.state = self._advance_batch(self.state, direction, moving & active)
        self.elapsed += active
        success = self.success(self.state, self.goal)
        terminated = self.transition.terminated | (success & active)
        truncated = self.transition.truncated | ((self.elapsed >= self.max_episode_steps) & ~terminated)
        self.transition = Transition(torch.where(active, success.float() - 0.01, 0), terminated, truncated, success)
        return self.render()

    def position(self, states):
        return states[..., 4:6] if self.task == "sokoban" else states[..., :2]

    object_position = position

    def distance(self, states, goal):
        goal = goal.view(len(goal), *([1] * (states.ndim - 2)), -1)
        d = (self.position(states) - self.position(goal)).abs().sum(-1)
        if self.task == "key_door":
            d = d + (states[..., 2] - goal[..., 2]).abs()
        if self.task == "checkpoints":
            d = d + (states[..., 3] - goal[..., 3]).abs()
        return d

    def success(self, states, goal):
        return self.distance(states, goal) == 0

    def tasks(self, generator=None):
        self.reset(generator=generator)
        return self.state.clone(), self.goal.clone()

    def _render_at(self, state, size):
        from torch_world.envs.puzzle_rendering import render_puzzle

        state = self.state if state is None else torch.as_tensor(state, device=self.device)
        # Bound temporary sprite buffers during offline dataset enumeration.
        return torch.cat([render_puzzle(self, chunk, size) for chunk in state.split(128)])

    def render(self, state=None):
        return self._render_at(state, self.size)

    def render_frame(self, state=None):
        return self._render_at(state, self.render_size)

    def render_goal_frame(self):
        return self.render_frame(self.goal)
