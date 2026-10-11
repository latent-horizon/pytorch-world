import hashlib
import zipfile

import pytest
import torch

from torch_world.datasets import MedicalVQA, MedMNIST, medmnist


def _npy(t: torch.Tensor, descr: str) -> bytes:
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {tuple(t.shape)}, }}".encode("latin1")
    header += b" " * (-(10 + len(header) + 1) % 64) + b"\n"
    return (
        b"\x93NUMPY\x01\x00"
        + len(header).to_bytes(2, "little")
        + header
        + bytes(t.contiguous().view(torch.uint8).flatten().tolist())
    )


@pytest.fixture
def root(tmp_path, monkeypatch):
    def write(name, n, gray=False, classes=3):
        shape = (n, 4, 4) if gray else (n, 4, 4, 3)
        path = tmp_path / f"{name}_64.npz"
        with zipfile.ZipFile(path, "w") as z:
            for split in ("train", "test"):
                z.writestr(f"{split}_images.npy", _npy(torch.randint(0, 255, shape, dtype=torch.uint8), "|u1"))
                z.writestr(f"{split}_labels.npy", _npy(torch.arange(n)[:, None] % classes, "<i8"))
        monkeypatch.setitem(medmnist.MD5, name, [None, hashlib.md5(path.read_bytes()).hexdigest(), None, None])

    write("bloodmnist", 6, classes=8)
    write("breastmnist", 2, gray=True, classes=2)
    return tmp_path


def test_medmnist_reads_npz_without_numpy(root):
    data = MedMNIST("breastmnist", "train", root=root, download=False)
    assert data.images.shape == (2, 3, 4, 4) and data.images.dtype == torch.uint8
    assert data.labels.tolist() == [0, 1] and data[1]["label"] == 1
    medmnist.MD5["breastmnist"][1] = "0" * 32
    with pytest.raises(RuntimeError, match="corrupt"):
        MedMNIST("breastmnist", root=root, download=False)


def test_medical_vqa_templates(root):
    vqa = MedicalVQA(["bloodmnist", "breastmnist"], root=root, download=False)
    kinds = [kind for *_, kind, _, _ in vqa.items]
    assert kinds.count("modality") == 8 and kinds.count("finding") == 6 and kinds.count("malignancy") == 2
    item = vqa[1]
    assert item["question"] == "what finding is shown ?" and item["answer"] == "basophil"
    assert item["caption"] == "blood smear microscopy showing a basophil" and item["image"].shape == (3, 4, 4)
    assert {"yes", "no", "platelet", "breast ultrasound"} <= set(vqa.answers)
    assert all(w in " ".join(vqa.texts).split() for w in ["is", "this", "an", "eosinophil", "?"])
    weights = vqa.weights()
    total = [weights[[i for i, (img, *_) in enumerate(vqa.items) if vqa.source[img] == k]].sum() for k in (0, 1)]
    assert torch.allclose(torch.stack(total), torch.ones(2))
    encoder = torch.nn.Linear(48, 5)
    vqa.encode(lambda x: encoder(x.flatten(1).float()), dtype=torch.float32, device="cpu")
    assert vqa[0]["image"].shape == (5,) and vqa.pixels([0, 1]).shape == (2, 3, 4, 4)
    assert "bloodmnist" in vqa.info and len(vqa.examples(2)[1]) == 4
