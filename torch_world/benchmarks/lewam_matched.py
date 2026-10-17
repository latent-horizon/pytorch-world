"""Matched from-scratch upstream/local LeWAM training on owned-arm data.

Each epoch visits every eligible start once. Both implementations share data,
initial weights, normalization, batches, goal sampling, LR and update count.
Their independent stochastic implementations need not consume identical RNG
streams. The upstream training loss is executed from the supplied checkout.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from torch_world.benchmarks.lewam_arm import ArmConfig, make_model, training_loss, windows
from torch_world.benchmarks.lewam_goal_reaching import goal_reaching_config
from torch_world.benchmarks.lewam_reference import ReferenceLeWAM, initialize_local_from_reference


def epoch_indices(episodes, steps, frameskip, seed):
    """One shuffled pass over all valid starts, including padded history/tails."""
    count = steps - frameskip + 1
    order = torch.randperm(episodes * count, generator=torch.Generator().manual_seed(seed))
    return order // count, order % count


def lr_scale(epoch, epochs, warmup=5):
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def float_latent_boundary(models):
    """Keep indexed transformer writes/SIGReg in fp32 under autocast, for both models.

    Upstream's token allocation follows encoder dtype, but indexed writes add
    fp32 position embeddings. PyTorch 2.14 rejects that mixed-dtype assignment.
    Casting at this shared boundary leaves encoder gradients attached.
    """

    def cast(module, inputs, output):
        return output.float()

    return [
        models["local"].projector.register_forward_hook(cast),
        models["upstream"].model.encoder.register_forward_hook(cast),
    ]


def train_matched(
    output, data_path, reference, *, epochs=50, batch_size=128, device="cuda", resume=False, precision="bf16"
):
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if precision == "bf16" and not str(device).startswith("cuda"):
        raise ValueError("bf16 study execution requires CUDA")
    if epochs < 1 or batch_size < 2:
        raise ValueError("Positive epochs and batch size >=2 required")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    output, data_path = Path(output), Path(data_path)
    output.mkdir(parents=True, exist_ok=resume)
    saved = ArmConfig(**json.loads((data_path.parent / "config.json").read_text()))
    config = goal_reaching_config(batch_size=batch_size, diverse_motion=saved.diverse_motion)
    for key in (
        "seed",
        "size",
        "steps",
        "train_episodes",
        "validation_episodes",
        "expert_gain",
        "goal_marker",
        "recovery_fraction",
        "diverse_motion",
    ):
        if getattr(config, key) != getattr(saved, key):
            raise ValueError(f"Dataset disagrees on {key}")
    data = torch.load(data_path, weights_only=True)
    flat = data["train"]["actions"].flatten(0, 1)
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    torch.manual_seed(config.seed)
    upstream = ReferenceLeWAM(config, reference, mean, std)
    ours = make_model(config, action_mean=mean, action_std=std)
    initialize_local_from_reference(ours, upstream)
    models = {"upstream": upstream.to(device), "local": ours.to(device)}
    if precision == "bf16":
        float_latent_boundary(models)
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4) for name, model in models.items()
    }
    record = {
        "protocol": "lewam-owned-arm-matched-v1",
        "config": asdict(config),
        "epochs": epochs,
        "batch_size": batch_size,
        "warmup_epochs": 5,
        "precision": precision,
        "latent_boundary": "fp32 for both models",
        "initialization": "identical active weights, from scratch",
        "data_path": str(data_path.resolve()),
        "reference_source_sha256": upstream.source_hashes,
        "host": platform.node(),
        "torch_version": torch.__version__,
        "curve": [],
        "selected_epochs": {},
        "best_validation": {"upstream": float("inf"), "local": float("inf")},
    }
    start_epoch = 0
    if resume:
        last = torch.load(output / "resume.pt", map_location=device, weights_only=True)
        record = last.get("record", json.loads((output / "training.json").read_text()))
        if len(record["curve"]) != last["epoch"]:
            raise ValueError("Checkpoint and training record disagree; preserve the run for recovery")
        if record["epochs"] != epochs or record["batch_size"] != batch_size or record["precision"] != precision:
            raise ValueError("Resume schedule differs")
        for name, model in models.items():
            model.load_state_dict(last["models"][name])
            optimizers[name].load_state_dict(last["optimizers"][name])
        start_epoch = last["epoch"]
    else:
        for name, model in models.items():
            torch.save({"model": model.state_dict(), "config": asdict(config)}, output / f"initial-{name}.pt")
        source = output / "source"
        source.mkdir()
        for filename in ("lewam_matched.py", "lewam_reference.py", "lewam_arm.py"):
            (source / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
        # Complete experiment source is also copied by the launch procedure.
        save_json(output / "training.json", record)
    # uint8 frames are resident on the GPU; no repeated 224px float transfer per step.
    data = {split: {k: v.to(device) for k, v in value.items()} for split, value in data.items()}
    val = windows(data["validation"], config, batch_size, torch.Generator().manual_seed(1234))
    val = {k: v.to(device) for k, v in val.items()}
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        ids, starts = epoch_indices(config.train_episodes, config.steps, config.frameskip, config.seed + epoch)
        totals = {name: {"loss": 0.0, "action_loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0} for name in models}
        for name, model in models.items():
            model.train()
            optimizers[name].param_groups[0]["lr"] = 1.5e-4 * lr_scale(epoch, epochs)
        seen = 0
        for begin in range(0, len(ids), batch_size):
            count = min(batch_size, len(ids) - begin)
            if count == 1:
                raise ValueError("BatchNorm requires more than one sample in the last batch")
            batch = windows(
                data["train"],
                config,
                count,
                torch.Generator().manual_seed(config.seed + epoch * 100000 + begin),
                episode_ids=ids[begin : begin + count],
                start_steps=starts[begin : begin + count],
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            for name, model in models.items():
                # Pair the seeds without pretending different layouts consume randomness identically.
                torch.manual_seed(config.seed + epoch * 100000 + begin)
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                    losses = model.loss(batch) if name == "upstream" else training_loss(model, batch)
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError(f"Nonfinite {name} loss at epoch {epoch + 1}")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                for key, value in losses.items():
                    totals[name][key] += float(value.detach() if torch.is_tensor(value) else value) * count
            seen += count
            if (begin // batch_size + 1) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "progress_epoch": epoch + 1,
                            "batches": begin // batch_size + 1,
                            "total_batches": math.ceil(len(ids) / batch_size),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        row = {
            "epoch": epoch + 1,
            "updates_per_model": (epoch + 1) * math.ceil(len(ids) / batch_size),
            "elapsed_seconds": time.perf_counter() - started,
        }
        for name, model in models.items():
            model.eval()
            torch.manual_seed(1234)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                losses = model.loss(val) if name == "upstream" else training_loss(model, val)
            score = float(losses["action_loss"])
            row[name] = {**{k: v / seen for k, v in totals[name].items()}, "validation_flow": score}
            if score < record["best_validation"][name]:
                record["best_validation"][name] = score
                record["selected_epochs"][name] = epoch + 1
                torch.save(
                    {"model": model.state_dict(), "config": asdict(config), "epoch": epoch + 1}, output / f"{name}.pt"
                )
        record["curve"].append(row)
        torch.save(
            {
                "epoch": epoch + 1,
                "record": record,
                "models": {k: v.state_dict() for k, v in models.items()},
                "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
            },
            output / "resume.tmp",
        )
        (output / "resume.tmp").replace(output / "resume.pt")
        save_json(output / "training.json", record)
        print(json.dumps(row), flush=True)
    record["checkpoint_sha256"] = {
        name: hashlib.sha256((output / f"{name}.pt").read_bytes()).hexdigest() for name in models
    }
    record["complete"] = True
    save_json(output / "training.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    a = p.parse_args()
    train_matched(
        a.output,
        a.data,
        a.reference,
        epochs=a.epochs,
        batch_size=a.batch_size,
        device=a.device,
        resume=a.resume,
        precision=a.precision,
    )


if __name__ == "__main__":
    main()
