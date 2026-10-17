"""Batched sprite rendering with cached geometry; no image or plotting dependency."""

import math

import torch
from torch.nn import functional as F


def _atlas(size, device, ice):
    # Supersampling is confined to the small shared atlas, never the state batch.
    n = size * 2
    axis = (torch.arange(n, device=device) + 0.5) / n * 2 - 1
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    tiles = torch.empty(13, n, n, 3, device=device)
    tiles[:] = torch.tensor([0.91, 0.96, 0.99] if ice else [0.95, 0.95, 0.92], device=device)

    def paint(ids, mask, color):
        for index in ids:  # thirteen shared sprites, independent of environment batch size
            tiles[index, mask] = torch.tensor(color, device=device)

    edge = (x.abs() > 0.96) | (y.abs() > 0.96)
    paint(range(13), edge, (0.85, 0.89, 0.91))
    paint([1], (x.abs() < 0.94) & (y.abs() < 0.94), (0.12, 0.17, 0.24))
    paint([1], (x.abs() < 0.82) & (y > -0.82) & (y < -0.65), (0.22, 0.29, 0.38))
    radius = (x.square() + y.square()).sqrt()
    paint([2, 9, 11], (radius < 0.86) & (radius > 0.64), (0.12, 0.62, 0.38))
    paint([2], (x.abs() < 0.09) | (y.abs() < 0.09), (0.12, 0.62, 0.38))
    key_ring = ((x + 0.28).square() + (y + 0.12).square()).sqrt()
    key = ((key_ring < 0.34) & (key_ring > 0.17)) | ((x > -0.03) & (x < 0.7) & ((y + 0.12).abs() < 0.10))
    key |= (x > 0.35) & (x < 0.65) & (y > -0.13) & (y < 0.24)
    paint([3], key, (0.83, 0.53, 0.02))
    paint([4], (x.abs() < 0.76) & (y.abs() < 0.90), (0.49, 0.29, 0.15))
    paint([4], (x.abs() < 0.60) & (y.abs() < 0.73), (0.73, 0.49, 0.25))
    paint([4], (radius < 0.19) | ((x.abs() < 0.09) & (y > 0) & (y < 0.39)), (0.15, 0.17, 0.20))
    for index, number in [(5, 1), (6, 2), (7, 1), (12, 2)]:
        paint([index], radius < 0.75, (0.52, 0.36, 0.73) if index < 7 else (0.25, 0.61, 0.43))
        digit = (x.abs() < 0.09) & (y.abs() < 0.44)
        if number == 2:
            digit = ((y.abs() < 0.09) | ((y.abs() - 0.4).abs() < 0.09)) & (x.abs() < 0.35)
            digit |= ((x - 0.27).abs() < 0.09) & (y > -0.4) & (y < 0)
            digit |= ((x + 0.27).abs() < 0.09) & (y > 0) & (y < 0.4)
        paint([index], digit, (1.0, 1.0, 1.0))
    box = (x.abs() < 0.61) & (y.abs() < 0.61)
    paint([8, 9], box, (0.52, 0.25, 0.08))
    paint([8, 9], (x.abs() < 0.50) & (y.abs() < 0.50), (0.92, 0.55, 0.22))
    paint([8, 9], box & (((x - y).abs() < 0.07) | ((x + y).abs() < 0.07)), (0.66, 0.32, 0.10))
    paint([10, 11], radius < 0.53, (0.08, 0.28, 0.49))
    paint([10, 11], radius < 0.43, (0.14, 0.48, 0.86))
    eyes = (((x - 0.15).square() + (y + 0.10).square()) < 0.006) | (((x + 0.15).square() + (y + 0.10).square()) < 0.006)
    paint([10, 11], eyes, (1.0, 1.0, 1.0))
    return F.avg_pool2d(tiles.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)


def render_puzzle(env, state, size):
    """Assemble cached sprites by index, preserving square cells and object silhouettes."""
    side = max(env.width, env.height)
    tile = max(2, math.ceil(size / side))
    if tile not in env._sprite_cache:
        env._sprite_cache[tile] = _atlas(tile, env.device, env.task == "ice")
    ids = env._wall_mask.long().expand(len(state), -1, -1).clone()
    batch = torch.arange(len(state), device=env.device)
    ids[:, env.target[1], env.target[0]] = 2
    if env.task == "key_door":
        locked = batch[state[:, 2] == 0]
        ids[locked, env.key[1], env.key[0]] = 3
        ids[locked, env.door[1], env.door[0]] = 4
    for j, (x, y) in enumerate(env.checkpoints):
        ids[:, y, x] = torch.where(state[:, 3] <= j, 5 + j, 7 if j == 0 else 12)
    if env.task == "sokoban":
        xy = state[:, 4:6].long()
        on_goal = (xy == torch.tensor(env.target, device=env.device)).all(-1)
        ids[batch, xy[:, 1], xy[:, 0]] = 8 + on_goal.long()
    xy = state[:, :2].long()
    ids[batch, xy[:, 1], xy[:, 0]] = 10 + (ids[batch, xy[:, 1], xy[:, 0]] == 2).long()
    pixels = (
        env._sprite_cache[tile][ids]
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(len(state), 3, env.height * tile, env.width * tile)
    )
    dx, dy = (side - env.width) * tile, (side - env.height) * tile
    pixels = F.pad(pixels, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2), value=0.975)
    return F.interpolate(pixels, (size, size), mode="bilinear", align_corners=False, antialias=True)
