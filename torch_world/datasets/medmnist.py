import ast
import hashlib
import random
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

# name: (imaging modality, class names, malignant class ids)
INFO = {
    "bloodmnist": (
        "blood smear microscopy",
        ["basophil", "eosinophil", "erythroblast", "immature granulocyte"]
        + ["lymphocyte", "monocyte", "neutrophil", "platelet"],
        set(),
    ),
    "breastmnist": ("breast ultrasound", ["malignant lesion", "benign lesion"], {0}),
    "dermamnist": (
        "dermatoscopy",
        ["actinic keratosis", "basal cell carcinoma", "benign keratosis", "dermatofibroma"]
        + ["melanoma", "melanocytic nevus", "vascular lesion"],
        {0, 1, 4},
    ),
    "organamnist": (
        "abdominal ct",
        ["bladder", "left femur", "right femur", "heart", "left kidney", "right kidney"]
        + ["liver", "left lung", "right lung", "pancreas", "spleen"],
        set(),
    ),
    "pathmnist": (
        "colon pathology",
        ["adipose tissue", "background", "debris", "lymphocytes", "mucus", "smooth muscle"]
        + ["normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"],
        {8},
    ),
}
MD5 = {
    "bloodmnist": ["7053d0359d879ad8a5505303e11de1dc", "2b94928a2ae4916078ca51e05b6b800b"]
    + ["adace1e0ed228fccda1f39692059dd4c", "b718ff6835fcbdb22ba9eacccd7b2601"],
    "breastmnist": ["750601b1f35ba3300ea97c75c52ff8f6", "742edef2a1fd1524b2efff4bd7ba9364"]
    + ["363e4b3f8d712e9b5de15470a2aaadf1", "b56378a6eefa9fed602bb16d192d4c8b"],
    "dermamnist": ["0744692d530f8e62ec473284d019b0c7", "b70a2f5635c6199aeaa28c31d7202e1f"]
    + ["2defd784463fa5243564e855ed717de1", "8974907d8e169bef5f5b96bc506ae45d"],
    "organamnist": ["68e3f8846a6bd62f0c9bf841c0d9eacc", "2dcccc29b88e6da5a01161ef20cda288"]
    + ["eeae80d0a227a8d099027e1b3cfd3b60", "50747347e05c87dd3aaf92c49f9f3170"],
    "pathmnist": ["a8b06965200029087d5bd730944a56c1", "55aa9c1e0525abe5a6b9d8343a507616"]
    + ["ac42d08fb904d92c244187169d1fd1d9", "2c51a510bcdc9cf8ddb2af93af1eadec"],
}
SIZES = (28, 64, 128, 224)
URL = "https://zenodo.org/records/10519652/files/{file}?download=1"
DTYPES = {"|u1": torch.uint8, "<u1": torch.uint8, "<i8": torch.int64, "<i4": torch.int32}


class MedMNIST(Dataset):
    """A MedMNIST v2 dataset: ``images`` ``(N, 3, size, size)`` uint8 and ``labels`` ``(N,)``.

    The ``.npz`` file is downloaded once from Zenodo into ``root`` and checked against its MD5. Grayscale images are
    repeated to three channels. Items are ``{"image", "label"}``. The data is licensed CC BY 4.0.

    Yang et al., "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification",
    Scientific Data 2023.
    Paper: https://arxiv.org/abs/2110.14795
    Data: https://zenodo.org/records/10519652
    """

    def __init__(
        self, name: str, split: str = "train", size: int = 64, root: str | Path | None = None, download: bool = True
    ):
        self.name, self.split, self.size = name, split, size
        self.modality, self.classes, self.malignant = INFO[name]
        arrays = _read_npz(self._fetch(Path(root or Path.home() / ".cache" / "torch_world" / "medmnist"), download))
        images = arrays[f"{split}_images"]
        images = images[..., None].expand(*images.shape, 3) if images.dim() == 3 else images
        self.images = images.permute(0, 3, 1, 2).contiguous()
        self.labels = arrays[f"{split}_labels"].flatten().long()

    def _fetch(self, root: Path, download: bool) -> Path:
        file = f"{self.name}.npz" if self.size == 28 else f"{self.name}_{self.size}.npz"
        path = root / file
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            root.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL.format(file=file), path)
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if md5 != MD5[self.name][SIZES.index(self.size)]:
            raise RuntimeError(f"{path} is corrupt (md5 {md5}); delete it and download again")
        return path

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {"image": self.images[i], "label": self.labels[i]}

    @property
    def info(self) -> str:
        return f"{self.name} ({self.split}): {self.modality}, {len(self.classes)} classes, {len(self)} images"


class MedicalVQA(Dataset):
    """Templated visual question answering over MedMNIST datasets, in the style of the MMBU challenge.

    Every image yields a **modality** question; datasets with more than two classes add a **finding** question and a
    **verification** of a named finding (true half of the time, drawn with ``seed``); datasets with malignant classes
    add a **malignancy** question. Items are ``{"image", "caption", "question", "answer", "kind"}``.

    ``images`` holds every image once as uint8. After ``encode(encoder)``, items carry the encoder's cached outputs in
    place of the images, so a frozen encoder runs once per image instead of once per question.

    MMBU challenge: https://akiranishii.github.io/mmbu-challenge/
    """

    KINDS = ("modality", "finding", "verification", "malignancy")

    def __init__(
        self,
        names: list[str],
        split: str = "train",
        size: int = 64,
        root: str | Path | None = None,
        download: bool = True,
        seed: int = 0,
    ):
        self.sources = [MedMNIST(name, split, size, root, download) for name in names]
        self.split = split
        self.images: Tensor = torch.cat([s.images for s in self.sources])
        self.features: Tensor | None = None
        self.source = torch.cat([torch.full((len(s),), k) for k, s in enumerate(self.sources)])
        self.captions, self.items = [], []
        rng = random.Random(seed)
        for k, s in enumerate(self.sources):
            for label in s.labels.tolist():
                image = len(self.captions)
                self.captions.append(caption(s, label))
                self.items += [(image, *qa) for qa in questions(s, label, rng)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        image, kind, question, answer = self.items[i]
        return {
            "image": (self.images if self.features is None else self.features)[image],
            "caption": self.captions[image],
            "question": question,
            "answer": answer,
            "kind": kind,
        }

    @property
    def info(self) -> str:
        """A table of the source datasets and of the questions of each kind."""
        counts = [[0] * len(self.KINDS) for _ in self.sources]
        for image, kind, *_ in self.items:
            counts[self.source[image]][self.KINDS.index(kind)] += 1
        head = f"{'dataset':13s}{'modality':24s}{'classes':>8s}{'images':>8s}" + "".join(
            f"{k:>14s}" for k in self.KINDS
        )
        rows = [
            f"{s.name:13s}{s.modality:24s}{len(s.classes):8d}{len(s):8d}" + "".join(f"{n:14d}" for n in c)
            for s, c in zip(self.sources, counts)
        ]
        total = f"{len(self.images)} images and {len(self)} questions ({self.split})"
        return "\n".join([head, *rows, total])

    def examples(self, per_source: int = 4) -> tuple[Tensor, list[str]]:
        """The first ``per_source`` images of each source dataset and their findings, for ``plots.images``."""
        picks = [(s, i) for s in self.sources for i in range(min(per_source, len(s)))]
        return torch.stack([s.images[i] for s, i in picks]), [s.classes[s.labels[i]] for s, i in picks]

    def pixels(self, items: list[int]) -> Tensor:
        """The uint8 images of the given items."""
        return self.images[[self.items[i][0] for i in items]]

    @torch.no_grad()
    def encode(self, encoder, batch_size: int = 512, dtype=torch.float16, device=None) -> "MedicalVQA":
        """Cache ``encoder(images)``, run on ``device`` (by default the encoder's); items then carry the outputs."""
        device = device or next(encoder.parameters()).device
        self.features = torch.cat([encoder(x.to(device)).to(dtype) for x in self.images.split(batch_size)])
        return self

    @property
    def texts(self) -> list[str]:
        """Every caption, question and answer the templates can produce for these datasets."""
        out = []
        for s in self.sources:
            for label in range(len(s.classes)):
                out.append(caption(s, label))
                out += [f"{q} {a}" for _, q, a in questions(s, label, None)]
        return out

    @property
    def answers(self) -> list[str]:
        """The candidate answers for retrieval: every answer the templates can produce."""
        return sorted({a for s in self.sources for label in range(len(s.classes)) for *_, a in questions(s, label)})

    def weights(self) -> Tensor:
        """Per-item sampling weights that give each source dataset the same total weight."""
        source = self.source[[image for image, *_ in self.items]]
        return 1.0 / source.bincount()[source]


def caption(source: MedMNIST, label: int) -> str:
    return f"{source.modality} showing {_article(source.classes[label])}"


def questions(source: MedMNIST, label: int, rng: random.Random | None = None) -> list[tuple[str, str, str]]:
    """``(kind, question, answer)`` for one image; without ``rng``, verification asks about every class."""
    out = [("modality", "what imaging modality is this ?", source.modality)]
    if len(source.classes) > 2:
        out.append(("finding", "what finding is shown ?", source.classes[label]))
        claims = range(len(source.classes)) if rng is None else [label if rng.random() < 0.5 else None]
        for claim in claims:
            claim = rng.randrange(len(source.classes)) if claim is None else claim
            answer = "yes" if claim == label else "no"
            out.append(("verification", f"is this {_article(source.classes[claim])} ?", answer))
    if source.malignant:
        out.append(("malignancy", "is the lesion malignant ?", "yes" if label in source.malignant else "no"))
    return out


def _article(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def _read_npz(path: Path) -> dict[str, Tensor]:
    """Read the arrays of an ``.npz`` file without NumPy."""
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            buffer = archive.read(name)
            width = 2 if buffer[6] == 1 else 4
            length = int.from_bytes(buffer[8 : 8 + width], "little")
            header = ast.literal_eval(buffer[8 + width : 8 + width + length].decode("latin1"))
            if header["fortran_order"]:
                raise ValueError(f"{name}: Fortran-ordered arrays are not supported")
            data = bytearray(buffer[8 + width + length :])
            arrays[name.removesuffix(".npy")] = torch.frombuffer(data, dtype=DTYPES[header["descr"]]).reshape(
                header["shape"]
            )
    return arrays
