import pytest
import torch

from torch_world.envs import PointMass
from torch_world.evaluation import VQAResults, compare, goal_reaching, random_policy, vqa


def test_goal_reaching_scores_each_policy():
    env = PointMass(batch_size=4, size=8)
    start = torch.full((4, 2), 0.5)
    goal = torch.tensor([[0.7, 0.5]] * 4)
    results = goal_reaching(
        env, {"right": lambda o: torch.tensor([[1.0, 0.0]] * 4), "random": random_policy(env)}, start, goal, 2
    )
    assert results.paths["right"].shape == (4, 3, 2) and results.distances["right"].shape == (4, 3)
    assert results.rates()["right"] == 1.0 and "right" in repr(results)
    summary = results.summary()["right"]
    assert summary["success"].value == 1.0 and summary["success"].low > 0.4
    assert summary["reached"] == 1.0 and summary["steps"] == 1.0 and summary["progress"] == pytest.approx(1.0)
    assert summary["SPL"] == pytest.approx(1.0) and summary["smoothness"] == 0 and summary["ms / step"] >= 0
    assert results.actions["right"].shape == (4, 2, 2) and results.reached["right"].shape == (4, 3)
    assert results.intervals()["random"].n == 4


def test_vqa_results_tables():
    results = VQAResults(None, ["yes", "no", "a", "b"], ["yes", "yes", "a", "c"], ["v", "v", "f", "f"])
    assert results.accuracy() == {"v": 0.5, "f": 0.5, "all": 0.5}
    assert results.majority() == {"v": 1.0, "f": 0.5}
    assert results.scores("m")["v"] == {"m": 0.5, "majority answer": 1.0} and "all" in repr(results)
    assert compare({"a": results, "b": results})["f"] == {"a": 0.5, "b": 0.5, "majority answer": 0.5}


class _Echo:
    """Answers each question with the index of the image it is shown."""

    def embed_text(self, ids, mask):
        return torch.zeros(len(ids), 1)

    def retrieve(self, x, bank, ids, mask):
        return x


def test_vqa_shuffled_images_mix_the_whole_dataset():
    dataset = [{"image": torch.tensor(i), "question": "q", "answer": str(i), "kind": "k"} for i in range(64)]
    tokenizer = lambda texts: (torch.zeros(len(texts), 1), torch.ones(len(texts), 1))
    candidates = [str(i) for i in range(64)]
    assert vqa(_Echo(), dataset, tokenizer, candidates, batch_size=8).accuracy()["all"] == 1.0
    shuffled = vqa(_Echo(), dataset, tokenizer, candidates, 8, True, torch.Generator().manual_seed(0)).predictions
    assert sorted(shuffled, key=int) == candidates
    assert any(int(p) >= 8 for p in shuffled[:8])


def test_goal_reaching_scores_reached_at_any_step():
    env = PointMass(batch_size=2, size=8)
    start, goal = torch.tensor([[0.2, 0.5], [0.2, 0.5]]), torch.tensor([[0.4, 0.5], [0.4, 0.5]])
    overshoot = {"right": lambda obs: torch.tensor([[1.0, 0.0], [1.0, 0.0]])}  # passes the goal, ends 0.3 past it
    final = goal_reaching(env, overshoot, start, goal, steps=5)
    reached = goal_reaching(env, overshoot, start, goal, steps=5, success="reached")
    assert not final.success["right"].any() and reached.success["right"].all()
    assert reached.intervals()["right"].value == 1.0
    with pytest.raises(ValueError):
        goal_reaching(env, overshoot, start, goal, steps=1, success="best")
