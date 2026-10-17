"""Explicit visual quality presets, independent of simulator dynamics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderQuality:
    observation_size: int
    render_size: int
    samples: int
    shadows: bool


QUALITY_PRESETS = {
    "low": RenderQuality(32, 160, 0, False),
    "medium": RenderQuality(64, 320, 2, True),
    "high": RenderQuality(128, 512, 4, True),
}


def resolve_quality(quality):
    try:
        return QUALITY_PRESETS[quality]
    except (KeyError, TypeError):
        raise ValueError(f"quality must be one of {tuple(QUALITY_PRESETS)}") from None
