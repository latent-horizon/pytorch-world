"""Save, load and share trained models: safetensors weights, a ``config.json`` that rebuilds the model, a model card.

A checkpoint is a folder (or a Hugging Face Hub repository) with ``model.safetensors``, ``config.json`` and
``README.md``. ``config["spec"]`` describes how to build the model, as nested ``{"class", "kwargs"}`` dicts naming
classes of ``torch_world``, so loading never executes code from the checkpoint::

    spec = {
        "class": "LeWM",
        "kwargs": {
            "encoder": {"class": "CLSEncoder", "kwargs": {"img_size": 64, "patch_size": 8}},
            "action_dim": 2,
        },
    }
    hub.save_pretrained(model, "lewm-tworooms", spec, card=hub.model_card("lewm-tworooms", "A LeWM ..."))
    model = hub.from_pretrained("latenthorizon/lewm-pusht")  # or a local folder

A spec may name a constructor other than the class itself, as ``{"class": "PatchTokens", "constructor": "from_hub",
"kwargs": {...}}``. Hub access needs ``pip install torch-world[hub]``.
"""

import importlib
import json
from pathlib import Path

from torch import nn

NAMESPACES = ("torch_world.models", "torch_world.nn", "torch_world.losses", "torch_world.models.tdmpc")
"""Modules whose public names a spec may build; nothing else can be instantiated from a config."""


def build(spec):
    """Instantiate a spec: ``{"class", "kwargs", "constructor"?}`` dicts are built recursively, other values kept."""
    if isinstance(spec, list):
        return [build(v) for v in spec]
    if not isinstance(spec, dict):
        return spec
    if "class" not in spec:
        return {k: build(v) for k, v in spec.items()}
    name = spec["class"]
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        if name in module.__all__:
            target = getattr(module, name)
            break
    else:
        raise ValueError(f"{name!r} is not a public class of {', '.join(NAMESPACES)}")
    constructor = spec.get("constructor")
    if constructor is not None:
        if constructor.startswith("_"):
            raise ValueError(f"constructor {constructor!r} is private")
        target = getattr(target, constructor)
    return target(**{k: build(v) for k, v in spec.get("kwargs", {}).items()})


def save_pretrained(
    model: nn.Module, path, spec: dict, card: str | None = None, external: tuple[str, ...] = (), **config
) -> Path:
    """Write ``model.safetensors``, ``config.json`` (``spec`` plus any ``config`` fields) and ``README.md`` to ``path``.

    ``external`` names submodules that ``spec`` loads from elsewhere, such as a frozen encoder built with
    ``from_hub``: their weights are not stored, and ``from_pretrained`` keeps the ones the spec loads.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().contiguous().cpu()
        for k, v in model.state_dict().items()
        if not any(k.startswith(f"{m}.") for m in external)
    }
    save_file(_untie(state), path / "model.safetensors")
    config = {"spec": spec, **({"external": list(external)} if external else {}), **config}
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if card is not None:
        (path / "README.md").write_text(card)
    return path


def from_pretrained(name_or_path, map_location="cpu", revision: str | None = None) -> nn.Module:
    """Build the model of a local folder or Hub repository from its spec and load its weights, in eval mode.

    Every weight must be in the checkpoint, except those of the ``external`` submodules of its config.
    """
    from safetensors.torch import load_file

    folder = Path(name_or_path)
    if not folder.is_dir():
        from huggingface_hub import snapshot_download

        folder = Path(
            snapshot_download(
                str(name_or_path), revision=revision, allow_patterns=["*.json", "*.md", "model.safetensors"]
            )
        )
    config = json.loads((folder / "config.json").read_text())
    model = build(config["spec"])
    external = tuple(f"{m}." for m in config.get("external", ()))
    missing, unexpected = model.load_state_dict(
        load_file(folder / "model.safetensors", device=str(map_location)), strict=False
    )
    missing = [k for k in missing if not k.startswith(external)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match its spec: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.config = config
    return model.to(map_location).eval()


def push_to_hub(path, repo_id: str, private: bool = False, message: str = "Upload model") -> str:
    """Upload a folder written by ``save_pretrained`` to the Hub; returns the repository URL."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(path), repo_id=repo_id, commit_message=message)
    return f"https://huggingface.co/{repo_id}"


def model_card(
    name: str,
    description: str,
    usage: str | None = None,
    results: list[dict] | None = None,
    paper: str | None = None,
    tags: tuple[str, ...] = ("world-model", "planning"),
    details: dict[str, str] | None = None,
) -> str:
    """A Hub model card: front matter, a description, how to load it, ``details`` as a list and ``results`` as a table.

    ``results`` rows are dicts with the same keys, for example ``{"task": ..., "episodes": ..., "success": ...}``.
    Report only measured numbers.
    """
    lines = ["---", "license: apache-2.0", "library_name: torch-world", f"tags: [{', '.join(tags)}]", "---", ""]
    lines += [f"# {name}", "", description.strip(), ""]
    lines += [
        "## Usage",
        "",
        "```python",
        usage or f'from torch_world import hub\n\nmodel = hub.from_pretrained("{name}")',
        "```",
        "",
    ]
    if details or paper:
        lines += [f"- {k}: {v}" for k, v in (details or {}).items()]
        lines += [f"- Paper: {paper}"] if paper else []
        lines.append("")
    if results:
        columns = list(results[0])
        lines += ["## Results", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in results]
        lines.append("")
    lines += ["Trained with [torch-world](https://github.com/latent-horizon/pytorch-world).", ""]
    return "\n".join(lines)


def _untie(state: dict) -> dict:
    """safetensors refuses tensors that share memory; clone all but the first of each shared storage."""
    seen, out = set(), {}
    for k, v in state.items():
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), v.shape)
        out[k] = v.clone() if key in seen else v
        seen.add(key)
    return out
