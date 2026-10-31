import torch
from torch import Tensor

from torch_world.envs.point_mass import PointMass
from torch_world.envs.sdf import box, circle, grid, paint, shade, union

DISTRACTOR_COLORS = [(0.86, 0.2, 0.18), (0.95, 0.6, 0.1), (0.2, 0.62, 0.3), (0.6, 0.25, 0.7), (0.1, 0.6, 0.7)]


class Rooms:
    """A vertical wall at ``x = 0.5`` with one door, for the point-mass worlds; active when ``self.rooms`` is true.

    The wall is ``wall`` thick; the door spans ``door ± door_width / 2`` in ``y``. The agent, ``margin`` in radius,
    cannot enter the wall: moves are integrated in ``substeps`` and a substep into the wall is undone. ``tasks`` puts
    the start in one room and the goal in the other, so the way to the goal runs through the door.
    """

    rooms = False
    wall, door, door_width, margin, substeps = 0.06, 0.5, 0.2, 0.02, 4

    def blocked(self, state: Tensor) -> Tensor:
        """Whether positions ``(..., 2)`` are inside the wall (and not in the door)."""
        in_wall = (state[..., 0] - 0.5).abs() < self.wall / 2 + self.margin
        in_door = (state[..., 1] - self.door).abs() < self.door_width / 2 - self.margin
        return in_wall & ~in_door & self.rooms

    def room(self, state: Tensor) -> Tensor:
        """``0`` left of the wall and ``1`` right of it, for positions ``(..., 2)``."""
        return (state[..., 0] > 0.5).long()

    def step(self, action: Tensor) -> Tensor:
        delta = self.max_step * action.clamp(-1, 1).to(self.state) / self.substeps
        for _ in range(self.substeps):
            moved = (self.state + delta).clamp(self.low, self.high)
            self.state = torch.where(self.blocked(moved)[:, None], self.state, moved)
        return self.render()

    def free(self, n: int, room: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        """``n`` positions outside the wall, in ``room`` ``(n,)`` (``0`` or ``1``) when given."""
        out = torch.empty(n, 2)
        todo = torch.ones(n, dtype=torch.bool)
        while todo.any():
            candidate = self.low + (self.high - self.low) * torch.rand(n, 2, generator=generator)
            ok = ~self.blocked(candidate) & todo
            if room is not None:
                ok &= self.room(candidate) == room.cpu()
            out[ok], todo = candidate[ok], todo & ~ok
        return out.to(self.state.device)

    def reset(self, state: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        if state is None and self.rooms:
            state = self.free(self.batch_size, generator=generator)
        return super().reset(state, generator)

    def tasks(
        self, generator: torch.Generator | None = None, same_room: bool = False, min_distance: float = 0.0
    ) -> tuple[Tensor, Tensor]:
        """Start and goal states ``(B, 2)`` in opposite rooms, or in the same room with ``same_room``.

        Goals closer than ``min_distance`` to their start are drawn again, so no task is solved in a step or two.
        """
        room = torch.randint(2, (self.batch_size,), generator=generator)
        goal_room = room if same_room else 1 - room
        start, goal = self.free(self.batch_size, room, generator), self.free(self.batch_size, goal_room, generator)
        while (near := (goal - start).norm(dim=-1) < min_distance).any():
            goal[near] = self.free(int(near.sum()), goal_room[near.cpu()], generator)
        return start, goal

    def wall_sdf(self, x: Tensor, y: Tensor) -> Tensor:
        """Signed distance to the wall, on image coordinates ``x``, ``y`` in ``[0, 1]``."""
        top = (self.door - self.door_width / 2) / 2
        bottom = (1 + self.door + self.door_width / 2) / 2
        return union(
            box(x, y, (0.5, top), (self.wall / 2, top)),
            box(x, y, (0.5, bottom), (self.wall / 2, 1 - bottom)),
        )


class TwoRooms(Rooms, PointMass):
    """PLDM's Two-Rooms: a point mass in the unit square split by a vertical wall with one door.

    Actions are velocities in ``[-1, 1]^2`` scaled by ``max_step``; a move into the wall stops at it (see ``Rooms``).
    ``tasks`` puts the start in one room and the goal in the other, so the shortest path goes through the door: greedy
    planning toward the goal gets stuck against the wall. ``collect`` keeps each action with probability 0.8, so that
    some episodes cross the door.

    Two nuisances test robustness without changing the task. ``distractors`` colored discs drift through the image on
    their own, unaffected by the actions and by the wall; they move one ``max_step`` per step in a direction that keeps
    with probability 0.8. ``floor`` tints the floor toward an RGB color (a background shift). Both are drawn into every
    image, goal images included; the state, distances and success ignore them.

    Sobal et al., "Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models", 2025.
    Paper: https://arxiv.org/abs/2502.14819
    """

    env_id = "TwoRooms-v0"
    low, high = 0.05, 0.95
    rooms = True

    def __init__(
        self,
        batch_size: int = 1,
        size: int = 64,
        max_step: float = 0.1,
        sigma: float = 0.04,
        wall: float = 0.06,
        door: float = 0.5,
        door_width: float = 0.2,
        substeps: int = 4,
        distractors: int = 0,
        floor: tuple[float, float, float] | None = None,
        device=None,
    ):
        super().__init__(batch_size, size, max_step, sigma, device=device)
        self.wall, self.door, self.door_width, self.substeps, self.margin = wall, door, door_width, substeps, sigma / 2
        if distractors > len(DISTRACTOR_COLORS):
            raise ValueError(f"at most {len(DISTRACTOR_COLORS)} distractors, not {distractors}")
        self.floor = floor
        self.distractors = torch.rand(batch_size, distractors, 2, device=device)
        self.headings = torch.rand(batch_size, distractors, device=device) * 2 * torch.pi

    def reset(self, state: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        n = self.distractors.size(1)
        self.distractors = torch.rand(self.batch_size, n, 2, generator=generator).to(self.distractors)
        self.headings = (torch.rand(self.batch_size, n, generator=generator) * 2 * torch.pi).to(self.headings)
        return super().reset(state, generator)

    def step(self, action: Tensor) -> Tensor:
        if self.distractors.size(1):
            turn = torch.rand(self.headings.shape) >= 0.8
            new = torch.rand(self.headings.shape) * 2 * torch.pi
            self.headings = torch.where(turn.to(self.headings.device), new.to(self.headings), self.headings)
            move = self.max_step * torch.stack([self.headings.cos(), self.headings.sin()], -1)
            self.distractors = (self.distractors + move).clamp(0, 1)
        return super().step(action)

    def collect(self, num_steps: int, generator: torch.Generator | None = None, keep: float = 0.8):
        return super().collect(num_steps, generator, keep)

    def render(self, state: Tensor | None = None) -> Tensor:
        """Images ``(B, 3, size, size)`` in ``[0, 1]``: the point over a light floor, the wall in gray, and the
        distractors, if any, at their current positions (those of the first episode when ``state`` has another batch
        size)."""
        state = self.state if state is None else state
        x, y = grid(self.size, state.device)
        image = super().render(state)
        if self.floor is not None:
            agent = (
                (-((x - state[:, 0, None, None]).square() + (y - state[:, 1, None, None]).square()) / 2)
                .div(self.sigma**2)
                .exp()[:, None]
            )
            tint = torch.tensor(self.floor, device=image.device)[:, None, None] - image
            image = image + 0.6 * (1 - agent) * tint
        image = paint(image, self.wall_sdf(x, y), shade(y, (0.4, 0.4, 0.42), (0.3, 0.3, 0.32)), self.size)
        distractors = self.distractors.to(state.device)
        if len(distractors) != len(state):
            distractors = distractors[:1].expand(len(state), -1, -1)
        for i, color in enumerate(DISTRACTOR_COLORS[: distractors.size(1)]):
            center = distractors[:, i, :, None, None].unbind(1)
            image = paint(image, circle(x, y, center, 1.5 * self.sigma), color, self.size)
        return image
