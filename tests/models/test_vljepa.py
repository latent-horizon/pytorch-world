import torch
from torch import nn

from torch_world.losses import info_nce
from torch_world.models import VLJEPA
from torch_world.nn import Transformer


class TextEncoder(nn.Module):
    def __init__(self, vocab=50, dim=12):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)

    def forward(self, ids, mask=None):
        mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask
        return (self.embed(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)


def make_model():
    x_encoder = nn.Sequential(nn.Flatten(2), nn.Linear(16, 20))
    return VLJEPA(
        x_encoder,
        Transformer(16, 2, 2),
        nn.Embedding(50, 16),
        TextEncoder(),
        vision_dim=20,
        predictor_dim=16,
        text_dim=12,
        embed_dim=8,
    )


def test_x_encoder_frozen_and_kept_in_eval():
    model = make_model().train()
    assert not model.x_encoder.training
    assert all(not p.requires_grad for p in model.x_encoder.parameters())


def test_padding_does_not_change_prediction():
    model = make_model().eval()
    x, ids = torch.randn(2, 3, 4, 4), torch.randint(0, 50, (2, 3))
    padded = torch.cat([ids, torch.zeros(2, 2, dtype=torch.long)], dim=1)
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 2, dtype=torch.bool)
    assert torch.allclose(model.predict(x, ids), model.predict(x, padded, mask), atol=1e-5)
    assert model.predict(x).shape == (2, 8)


def test_loss_is_bidirectional_info_nce():
    model = make_model()
    x, target = torch.randn(4, 3, 4, 4), torch.randint(0, 50, (4, 5))
    with torch.no_grad():
        expected = info_nce(model.predict(x), model.embed_text(target), 0.07)
        assert torch.allclose(model(x, target)["loss"], expected, atol=1e-5)


def test_param_groups_scale_y_encoder_lr():
    model = make_model()
    groups = model.param_groups(1e-3, weight_decay=0.1)
    assert groups[1]["lr"] == 1e-3 * 0.05 and groups[0]["weight_decay"] == 0.1
    assert len(groups[1]["params"]) == len(list(model.y_encoder.parameters()))
    counted = sum(p.numel() for g in groups for p in g["params"])
    assert counted == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_segment_finds_piecewise_constant_segments():
    stream = torch.cat([torch.zeros(4, 3), torch.ones(6, 3), torch.full((3, 3), 5.0)])
    assert VLJEPA.segment(stream + 0.01 * torch.randn(13, 3), 3).tolist() == [1, 6, 11]


def test_retrieve_picks_the_closest_candidate():
    model = make_model().eval()
    x, query = torch.randn(3, 3, 4, 4), torch.randint(0, 50, (3, 2))
    pred = model.predict(x, query)
    bank = torch.cat([torch.randn(2, 8), pred[1:2] * 5])
    assert model.retrieve(x, bank, query)[1] == 2


def test_word_tokenizer_and_text_encoder():
    from torch_world.nn import TextEncoder as Encoder
    from torch_world.nn import WordTokenizer

    tokenizer = WordTokenizer(["is this a cell ?", "yes"])
    ids, mask = tokenizer(["yes", "is this a cell ?"])
    assert len(tokenizer) == 7 and ids.shape == (2, 5) and mask.sum().item() == 6
    assert Encoder(len(tokenizer), dim=16)(ids, mask).shape == (2, 16)
