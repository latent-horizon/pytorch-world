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
