import torch
from torch import Tensor

from torch_world.envs.point_mass import PointMass
from torch_world.envs.sdf import box, grid, paint, shade, union


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

    def tasks(self, generator: torch.Generator | None = None, same_room: bool = False) -> tuple[Tensor, Tensor]:
        """Start and goal states ``(B, 2)`` in opposite rooms, or in the same room with ``same_room``."""
        room = torch.randint(2, (self.batch_size,), generator=generator)
        goal_room = room if same_room else 1 - room
        return self.free(self.batch_size, room, generator), self.free(self.batch_size, goal_room, generator)

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
        device=None,
    ):
        super().__init__(batch_size, size, max_step, sigma, device=device)
        self.wall, self.door, self.door_width, self.substeps, self.margin = wall, door, door_width, substeps, sigma / 2

    def collect(self, num_steps: int, generator: torch.Generator | None = None, keep: float = 0.8):
        return super().collect(num_steps, generator, keep)

    def render(self, state: Tensor | None = None) -> Tensor:
        """Images ``(B, 3, size, size)`` in ``[0, 1]``: the point over a light floor, the wall in gray."""
        state = self.state if state is None else state
        x, y = grid(self.size, state.device)
        return paint(
            super().render(state), self.wall_sdf(x, y), shade(y, (0.4, 0.4, 0.42), (0.3, 0.3, 0.32)), self.size
        )
