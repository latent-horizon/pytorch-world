"""Distributed DreamZero: encode play data into latent shards, train on many GPUs, evaluate in closed loop.

Every command runs one process per GPU under ``torchrun`` (or ``srun`` launching ``torchrun`` on each node) and also
runs as a single plain process. The three steps share an output layout::

    torchrun --nproc-per-node 8 -m torch_world.benchmarks.dreamzero prepare --output runs/dz/data --episodes 8192
    torchrun --nproc-per-node 8 -m torch_world.benchmarks.dreamzero train --data runs/dz/data --output runs/dz/train
    torchrun --nproc-per-node 8 -m torch_world.benchmarks.dreamzero evaluate --run runs/dz/train --episodes 512

``prepare`` splits episodes across ranks; each rank collects ``RobotCube`` play data, encodes clips with the frozen
Wan 2.1 ``VideoVAE`` and writes its own shards, so encoding scales with the number of GPUs. ``train`` streams the
shards through a ``DistributedSampler`` and wraps ``DreamZero`` in DDP or FSDP2 (``--strategy fsdp`` shards every
transformer block and the optimizer state across ranks, for models that do not fit one GPU), with bf16, gradient
accumulation, warmup + cosine learning rate and resumable checkpoints. ``evaluate`` splits goal-reaching episodes
across ranks and gathers their success. Only ``prepare`` and ``evaluate`` know the environment; a dataset from any
other source only needs shards with ``latents``, ``actions`` and, for goal-conditioned models, ``goal``.
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from torch_world.datasets import LatentClips, write_latent_shard
from torch_world.models import DreamZero

MODEL_KEYS = (
    "patch_size",
    "chunk_frames",
    "chunk_actions",
    "num_chunks",
    "dim",
    "depth",
    "num_heads",
    "flow_steps",
    "goal_conditioned",
    "flash_alpha",
    "action_weight",
    "video_weight",
    "time_schedule",
    "attention",
)


# Distributed -----------------------------------------------------------------------------------------------------


def setup() -> tuple[int, int, torch.device]:
    """Join the process group when launched by ``torchrun``; returns ``(rank, world_size, device)``."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return (dist.get_rank() if dist.is_initialized() else 0), world, device


def teardown():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def mean_across_ranks(values: dict[str, float], device) -> dict[str, float]:
    if not dist.is_initialized():
        return values
    t = torch.tensor(list(values.values()), device=device, dtype=torch.float64)
    dist.all_reduce(t)
    return dict(zip(values, (t / dist.get_world_size()).tolist()))


def save_json(path: Path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


# Prepare ---------------------------------------------------------------------------------------------------------


def clip_geometry(chunk_frames: int, chunk_actions: int, num_chunks: int, temporal_ratio: int) -> tuple[int, int]:
    """``(frames, frame_stride)`` of a clip: environment steps between recorded frames, and frames per clip."""
    raw = chunk_frames * temporal_ratio
    if chunk_actions % raw:
        raise ValueError("chunk_actions must be a multiple of chunk_frames * temporal_ratio")
    return 1 + num_chunks * raw, chunk_actions // raw


def reach_demos(env, steps: int, generator: torch.Generator, noise: float, retarget: int) -> dict:
    """Noisy scripted reaches with the gripper open, toward a new random aim every ``retarget`` steps.

    Goal-directed demonstrations for the ``"reach"`` task, unlike play data: relabeling a clip's last frame as its goal
    pairs every goal with actions that move toward it.
    """
    pixels, actions = [env.reset(generator=generator)], []
    for t in range(steps):
        if t % retarget == 0:
            aim = env.reach_targets(generator)
        action = env.scripted(torch.cat([aim, env.state[:, 3:].to(aim)], -1)).cpu()
        action = (action + noise * torch.randn(action.shape, generator=generator)).clamp(-1, 1)
        action[:, 3] = 1.0
        actions.append(action)
        pixels.append(env.step(action))
    return {"pixels": torch.stack(pixels, 1), "actions": torch.stack(actions, 1)}


def prepare(args):
    from torch_world.envs import RobotCube
    from torch_world.nn import VideoVAE

    rank, world, device = setup()
    vae = VideoVAE.from_pretrained().to(device)
    frames, stride = clip_geometry(args.chunk_frames, args.chunk_actions, args.num_chunks, vae.temporal_ratio)
    span = args.num_chunks * args.chunk_actions
    output = Path(args.output)
    batches = math.ceil(args.episodes / args.env_batch)
    robot = {"robot": args.robot, "size": args.size, "control_dt": args.control_dt, "max_step": args.max_step}
    env = RobotCube(batch_size=args.env_batch, task="reach" if args.data == "reach" else "pick", **robot)
    for b in range(rank, batches, world):
        path = output / f"shard-{b:05d}.pt"
        if path.exists():
            continue
        generator = torch.Generator().manual_seed(args.seed * 100003 + b)
        if args.data == "reach":
            data = reach_demos(env, args.steps, generator, args.noise, args.retarget)
        else:
            data = env.collect(args.steps + 1, noise=args.noise, generator=generator)
        latents, goals, actions = [], [], []
        for s in range(0, args.steps - span + 1, args.clip_stride):
            clip = data["pixels"][:, s : s + span + 1 : stride]
            with torch.no_grad():
                latents.append(torch.cat([vae.encode(c.to(device)) for c in clip.split(args.encode_batch)]).cpu())
                goals.append(
                    torch.cat([vae.encode(c.to(device)) for c in clip[:, -1:].split(args.encode_batch)])[:, 0].cpu()
                )
            actions.append(data["actions"][:, s : s + span])
        assert latents[0].size(1) == 1 + args.num_chunks * args.chunk_frames, (latents[0].shape, frames)
        write_latent_shard(path, latents=torch.cat(latents), goal=torch.cat(goals), actions=torch.cat(actions))
        print(f"[rank {rank}] wrote {path.name}", flush=True)
    if rank == 0:
        config = {
            "latent_channels": vae.latent_channels,
            "latent_size": [args.size // vae.spatial_ratio] * 2,
            "action_dim": env.action_dim,
            "chunk_frames": args.chunk_frames,
            "chunk_actions": args.chunk_actions,
            "num_chunks": args.num_chunks,
            "robot": robot,
            "data": args.data,
        }
        save_json(output / "config.json", config)
    teardown()


# Train -----------------------------------------------------------------------------------------------------------


def build_model(config: dict, device, autoencoder=None) -> DreamZero:
    kwargs = {k: config[k] for k in MODEL_KEYS if k in config}
    return DreamZero(
        config["latent_channels"], tuple(config["latent_size"]), config["action_dim"], autoencoder=autoencoder, **kwargs
    ).to(device)


def wrap(model: DreamZero, strategy: str, device):
    """DDP replicates the model; FSDP2 shards every transformer block, then the rest, with bf16 compute."""
    if not dist.is_initialized():
        return model
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh(device.type, (dist.get_world_size(),))

    policy = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if device.type == "cuda"
        else MixedPrecisionPolicy()
    )
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    return fully_shard(model, mesh=mesh, mp_policy=policy)


def lr_at(step: int, steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))


def full_state(model, optimizer):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    return get_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def load_state(model, optimizer, checkpoint):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

    set_state_dict(
        model,
        optimizer,
        model_state_dict=checkpoint["model"],
        optim_state_dict=checkpoint["optimizer"],
        options=StateDictOptions(full_state_dict=True),
    )


@torch.no_grad()
def swapped_state(model, optimizer, weights, ema) -> dict:
    """The full model state with the EMA weights in place of the trained ones, which are then restored."""
    trained = [p.detach().clone() for p in weights]
    for p, e in zip(weights, ema):  # one by one: FSDP2's sharded tensors have no batched copy
        p.copy_(e)
    # Unsharded CPU states can alias the parameters, so copy before restoring them.
    state = {k: v.clone() for k, v in full_state(model, optimizer)[0].items()}
    for p, t in zip(weights, trained):
        p.copy_(t)
    return state


def train(args):
    rank, world, device = setup()
    torch.manual_seed(args.seed)
    data = LatentClips(args.data)
    config = json.loads((Path(args.data) / "config.json").read_text())
    config.update({k: getattr(args, k) for k in MODEL_KEYS if getattr(args, k, None) is not None})
    config["goal_conditioned"] = not args.no_goal
    model = build_model(config, device)
    params = sum(p.numel() for p in model.parameters())
    model = wrap(model, args.strategy, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    step = 0
    weights = list(model.parameters())
    ema = [p.detach().clone() for p in weights] if args.ema > 0 else None
    if (output / "checkpoint.pt").exists():
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        if ema is not None and "ema" in checkpoint:
            load_state(model, optimizer, {"model": checkpoint["ema"], "optimizer": checkpoint["optimizer"]})
            ema = [p.detach().clone() for p in weights]
        load_state(model, optimizer, checkpoint)
        step = checkpoint["step"]
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    per_epoch = len(loader) // args.accumulate
    if per_epoch < 1:
        raise ValueError("Each rank needs at least batch_size * accumulate clips")
    if rank == 0:
        save_json(output / "config.json", config)
        print(
            f"{params / 1e6:.1f}M parameters, {len(data)} clips, {world} ranks, "
            f"global batch {args.batch_size * args.accumulate * world}",
            flush=True,
        )
    autocast = (
        torch.autocast("cuda", torch.bfloat16) if device.type == "cuda" and args.strategy == "ddp" else nullcontext()
    )
    log = (output / "log.jsonl").open("a") if rank == 0 else None
    tic, epoch = time.time(), step // per_epoch
    while step < args.steps:
        sampler.set_epoch(epoch)
        batches = iter(loader)
        for _ in range(per_epoch):
            if step >= args.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_at(step, args.steps, args.warmup)
            totals = {"loss": 0.0, "video_loss": 0.0, "action_loss": 0.0}
            for micro in range(args.accumulate):
                batch = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
                sync = micro == args.accumulate - 1
                context = nullcontext()
                if dist.is_initialized() and args.strategy == "ddp" and not sync:
                    context = model.no_sync()
                elif dist.is_initialized() and args.strategy == "fsdp":
                    model.set_requires_gradient_sync(sync)
                with context, autocast:
                    out = model(batch["latents"], batch["actions"], batch.get("states"), goal=batch.get("goal"))
                    (out["loss"] / args.accumulate).backward()
                for k in totals:
                    totals[k] += out[k].item() / args.accumulate
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                decay = min(args.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    torch._foreach_lerp_(ema, [p.detach() for p in weights], 1 - decay)
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                totals = mean_across_ranks(
                    {**totals, "grad_norm": float(grad.full_tensor() if hasattr(grad, "full_tensor") else grad)}, device
                )
                if rank == 0:
                    record = {
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "seconds": time.time() - tic,
                        **totals,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                model_state, optimizer_state = full_state(model, optimizer)
                ema_state = model_state if ema is None else swapped_state(model, optimizer, weights, ema)
                if rank == 0:
                    torch.save(
                        {"model": model_state, "optimizer": optimizer_state, "ema": ema_state, "step": step},
                        output / "checkpoint.tmp",
                    )
                    (output / "checkpoint.tmp").replace(output / "checkpoint.pt")
                    torch.save(ema_state, output / "model.pt")
                if dist.is_initialized():
                    dist.barrier()
        epoch += 1
    teardown()


# Evaluate --------------------------------------------------------------------------------------------------------


def evaluate(args):
    from torch_world import evaluation, metrics
    from torch_world.envs import RobotCube
    from torch_world.nn import VideoVAE
    from torch_world.planning import DreamZeroPolicy, WAMPlanner

    rank, world, device = setup()
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    model = build_model(config, device, VideoVAE.from_pretrained().to(device))
    if args.flow_steps:
        model.flow_steps = args.flow_steps
    missing, unexpected = model.load_state_dict(
        torch.load(run / "model.pt", map_location=device, weights_only=True), strict=False
    )
    if unexpected or any(not k.startswith("autoencoder.") for k in missing):
        raise RuntimeError(f"Checkpoint does not match the model: {missing=} {unexpected=}")
    model.eval()
    env = RobotCube(batch_size=args.env_batch, task=args.task, **config["robot"])
    success = []
    for b in range(rank, math.ceil(args.episodes / args.env_batch), world):
        start, goal_state = env.tasks(generator=torch.Generator().manual_seed(args.seed * 100003 + b))
        goal = None
        if model.goal_conditioned:
            with torch.no_grad():
                goal = model.encode(env.render(goal_state)[:, None].to(device))[:, 0]
        plan = WAMPlanner(num_samples=args.samples) if args.samples > 1 else None
        policy = DreamZeroPolicy(model, goal, plan=plan, generator=torch.Generator(device).manual_seed(b))
        result = evaluation.goal_reaching(
            env, {"DreamZero": lambda o, policy=policy: policy(o.to(device)).cpu()}, start, goal_state, args.horizon
        )
        success.append(result.success["DreamZero"].cpu())
        print(f"[rank {rank}] batch {b}: {success[-1].float().mean():.2f}", flush=True)
    local = torch.cat(success) if success else torch.zeros(0, dtype=torch.bool)
    gathered = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank == 0:
        everything = torch.cat(gathered)[: args.episodes]
        rate = metrics.success_rate(everything)
        report = {
            "task": args.task,
            "episodes": everything.numel(),
            "success": rate.value,
            "low": rate.low,
            "high": rate.high,
            "samples": args.samples,
            "horizon": args.horizon,
        }
        save_json(run / f"evaluation-{args.task}.json", report)
        print(json.dumps(report), flush=True)
    teardown()


# Diagnose --------------------------------------------------------------------------------------------------------


@torch.no_grad()
def diagnose(args):
    """Open-loop errors on held-out clips, to tell a weak video model from a weak action branch.

    From an anchor frame and the first real chunk, the model predicts chunk 2. ``policy`` is the action half of joint
    sampling; ``inverse_dynamics`` denoises actions with the *true* chunk-2 video given clean, so the gap between the
    two is what imperfect imagined video costs. ``dynamics`` denoises chunk-2 video from the true actions, against
    repeating the last real latent frame. Mean squared errors in environment action units and normalized latents.
    """
    _, _, device = setup()
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    model = build_model(config, device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device, weights_only=True))
    model.eval()
    if args.flow_steps:
        model.flow_steps = args.flow_steps
    data = LatentClips(args.data)
    K, H = model.chunk_frames, model.chunk_size
    totals, count = {}, 0
    generator = torch.Generator(device).manual_seed(args.seed)
    subset = torch.utils.data.Subset(data, range(min(args.clips, len(data))))
    for batch in DataLoader(subset, batch_size=args.batch_size):
        z, actions = batch["latents"].to(device), batch["actions"].to(device)
        goal = batch["goal"].to(device) if model.goal_conditioned else None
        context, past = z[:, : 1 + K], actions[:, :H]
        true_actions, true_video = actions[:, H : 2 * H], z[:, 1 + K : 1 + 2 * K]
        kwargs = {"past_actions": past, "generator": generator}
        _, policy = model.imagine(context, goal, **kwargs)
        inverse = model.inverse_dynamics(context, true_video, goal, **kwargs)
        future = model.predict_future(context, true_actions, goal, **kwargs)
        errors = {
            "policy_action_mse": (policy - true_actions).square().mean((1, 2)),
            "inverse_dynamics_action_mse": (inverse - true_actions).square().mean((1, 2)),
            "zero_action_mse": true_actions.square().mean((1, 2)),
            "dynamics_latent_mse": (future - true_video).square().flatten(1).mean(1),
            "copy_last_latent_mse": (context[:, -1:] - true_video).square().flatten(1).mean(1),
        }
        for k, v in errors.items():
            totals[k] = totals.get(k, 0.0) + v.sum().item()
        count += z.size(0)
    report = {k: v / count for k, v in totals.items()} | {"clips": count, "flow_steps": model.flow_steps}
    save_json(run / "diagnose.json", report)
    print(json.dumps(report, indent=2), flush=True)
    teardown()


# CLI -------------------------------------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = p.add_subparsers(dest="command", required=True)

    pre = commands.add_parser("prepare", help="collect play data and write Wan VAE latent shards")
    pre.add_argument("--output", required=True)
    pre.add_argument("--episodes", type=int, default=2048)
    pre.add_argument("--steps", type=int, default=64, help="environment steps per episode")
    pre.add_argument("--clip-stride", type=int, default=8, help="steps between clip starts")
    pre.add_argument("--env-batch", type=int, default=256)
    pre.add_argument("--encode-batch", type=int, default=32)
    pre.add_argument("--robot", default="ur10e")
    pre.add_argument("--size", type=int, default=96)
    pre.add_argument("--control-dt", type=float, default=0.2)
    pre.add_argument("--max-step", type=float, default=0.1)
    pre.add_argument("--chunk-frames", type=int, default=2)
    pre.add_argument("--chunk-actions", type=int, default=8)
    pre.add_argument("--num-chunks", type=int, default=4)
    pre.add_argument("--data", choices=("play", "reach"), default="play", help="play data or noisy reach demos")
    pre.add_argument("--noise", type=float, default=0.04, help="action noise of the scripted policy")
    pre.add_argument("--retarget", type=int, default=12, help="steps between reach aims")
    pre.add_argument("--seed", type=int, default=0)

    tr = commands.add_parser("train", help="train DreamZero with DDP or FSDP2")
    tr.add_argument("--data", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--strategy", choices=("ddp", "fsdp"), default="ddp")
    tr.add_argument("--steps", type=int, default=20000)
    tr.add_argument("--batch-size", type=int, default=32, help="clips per rank per micro-batch")
    tr.add_argument("--accumulate", type=int, default=1)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight-decay", type=float, default=1e-3)
    tr.add_argument("--warmup", type=int, default=1000)
    tr.add_argument("--clip", type=float, default=1.0)
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--log-every", type=int, default=50)
    tr.add_argument("--save-every", type=int, default=1000)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--patch-size", type=int, default=2)
    tr.add_argument("--dim", type=int, default=768)
    tr.add_argument("--depth", type=int, default=12)
    tr.add_argument("--num-heads", type=int, default=12)
    tr.add_argument("--flow-steps", type=int, default=16)
    tr.add_argument("--flash-alpha", type=float, default=None)
    tr.add_argument("--action-weight", type=float, default=None)
    tr.add_argument("--video-weight", type=float, default=None, help="0 trains the same network as a pure policy")
    tr.add_argument("--time-schedule", choices=("uniform", "logit_normal"), default=None)
    tr.add_argument("--ema", type=float, default=0.999, help="decay of the weight average saved as model.pt; 0 is off")
    tr.add_argument("--attention", choices=("sdpa", "flex"), default="flex")
    tr.add_argument("--no-goal", action="store_true", help="train without goal conditioning")

    ev = commands.add_parser("evaluate", help="closed-loop goal reaching, split across ranks")
    ev.add_argument("--run", required=True, help="a train --output directory")
    ev.add_argument("--task", choices=("reach", "pick"), default="reach")
    ev.add_argument("--episodes", type=int, default=256)
    ev.add_argument("--env-batch", type=int, default=32)
    ev.add_argument("--horizon", type=int, default=16)
    ev.add_argument("--samples", type=int, default=1, help="proposals ranked by WAMPlanner; 1 is the policy")
    ev.add_argument("--flow-steps", type=int, default=None, help="override the integration steps")
    ev.add_argument("--seed", type=int, default=1)

    dg = commands.add_parser("diagnose", help="open-loop policy, inverse-dynamics and dynamics errors")
    dg.add_argument("--run", required=True)
    dg.add_argument("--data", required=True, help="held-out shards, from prepare with another --seed")
    dg.add_argument("--clips", type=int, default=512)
    dg.add_argument("--batch-size", type=int, default=64)
    dg.add_argument("--flow-steps", type=int, default=None)
    dg.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    {"prepare": prepare, "train": train, "evaluate": evaluate, "diagnose": diagnose}[args.command](args)


if __name__ == "__main__":
    main()
