import torch

from torch_world.envs import TwoRooms


def test_wall_blocks_and_door_lets_through():
    env = TwoRooms(batch_size=2, size=16)
    env.reset(torch.tensor([[0.3, 0.15], [0.3, 0.5]]))
    for _ in range(5):
        env.step(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert env.room(env.state).tolist() == [0, 1]
    assert not env.blocked(env.state).any()


def test_tasks_rooms_and_collect_shapes():
    env = TwoRooms(batch_size=64, size=16)
    g = torch.Generator().manual_seed(0)
    start, goal = env.tasks(g)
    assert (env.room(start) != env.room(goal)).all()
    start, goal = env.tasks(g, same_room=True)
    assert (env.room(start) == env.room(goal)).all()
    data = env.collect(12, generator=g)
    assert data["pixels"].shape == (64, 12, 3, 16, 16) and data["actions"].shape == (64, 12, 2)
    assert not env.blocked(data["states"]).any()
    assert env.success(goal, goal).all()


def test_macaw_rooms():
    from torch_world.envs import Macaw

    env = Macaw(batch_size=64, size=16, rooms=True)
    start, goal = env.tasks(torch.Generator().manual_seed(0))
    assert (env.room(start) != env.room(goal)).all()
    data = env.collect(12, generator=torch.Generator().manual_seed(0))
    assert not env.blocked(data["states"]).any()
    env.reset(torch.tensor([[0.3, 0.2]]).expand(64, -1))
    for _ in range(5):
        env.step(torch.tensor([[1.0, 0.0]]).expand(64, -1))
    assert (env.room(env.state) == 0).all()
    assert not Macaw(size=16).blocked(torch.tensor([[0.5, 0.2]])).any()


def test_distractors_and_floor_change_pixels_not_state():
    g = torch.Generator().manual_seed(0)
    clean = TwoRooms(batch_size=4, size=32)
    start = clean.free(4, generator=g)
    noisy = TwoRooms(batch_size=4, size=32, distractors=3, floor=(0.9, 0.5, 0.5))
    a, b = clean.reset(start), noisy.reset(start)
    assert not torch.allclose(a, b)
    before = noisy.distractors.clone()
    for _ in range(3):
        clean.step(torch.ones(4, 2))
        noisy.step(torch.ones(4, 2))
    assert torch.equal(clean.state, noisy.state)
    assert not torch.equal(before, noisy.distractors) and ((noisy.distractors >= 0) & (noisy.distractors <= 1)).all()
    assert noisy.render(start[:1]).shape == (1, 3, 32, 32)
    assert torch.equal(TwoRooms(batch_size=4, size=32).reset(start), a)
