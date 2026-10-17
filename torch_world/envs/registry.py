"""Lazy environment discovery and user-defined task registration."""

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from importlib import import_module
from inspect import signature
from types import MappingProxyType
from typing import Any

from torch_world.envs.benchmark import BenchmarkEnv
from torch_world.envs.legacy import LegacyEnvironment


@dataclass(frozen=True)
class EnvSpec:
    """A versioned task recipe. Listing specs never imports optional simulators."""

    id: str
    entry_point: str | Callable
    backend: str
    description: str
    tags: tuple[str, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    reference: str = ""
    category: str = ""

    def __post_init__(self):
        if not self.id or not self.backend or not self.description:
            raise ValueError("id, backend and description are required")
        if not callable(self.entry_point) and (not isinstance(self.entry_point, str) or ":" not in self.entry_point):
            raise ValueError("entry_point must be callable or 'module:attribute'")
        object.__setattr__(self, "kwargs", MappingProxyType(deepcopy(dict(self.kwargs))))
        object.__setattr__(self, "tags", tuple(self.tags))


_REGISTRY: dict[str, EnvSpec] = {}


def register_env(spec: EnvSpec) -> EnvSpec:
    """Register a recipe explicitly; duplicate IDs are errors, never silent overrides."""
    if spec.id in _REGISTRY:
        raise ValueError(f"Environment {spec.id!r} is already registered")
    _REGISTRY[spec.id] = spec
    return spec


def list_envs(
    *, backend: str | None = None, tag: str | None = None, category: str | None = None
) -> tuple[EnvSpec, ...]:
    """Sorted specs, optionally filtered by backend, a capability tag, and a category from `CATEGORIES`."""
    return tuple(
        s
        for _, s in sorted(_REGISTRY.items())
        if (backend is None or s.backend == backend)
        and (tag is None or tag in s.tags)
        and (category is None or s.category == category)
    )


def is_environment(value) -> bool:
    """Whether a value is an existing world rather than a constructor or recipe."""
    return isinstance(value, (BenchmarkEnv, LegacyEnvironment))


def get_spec(id: str | EnvSpec | type[BenchmarkEnv | LegacyEnvironment] | BenchmarkEnv | LegacyEnvironment) -> EnvSpec:
    """Resolve a world instance, registry ID, recipe, or named class to its metadata."""
    if is_environment(id):
        return get_spec(getattr(id, "spec", getattr(id, "env_id", None)))
    if isinstance(id, EnvSpec):
        return id
    if isinstance(id, type) and issubclass(id, (BenchmarkEnv, LegacyEnvironment)):
        id = getattr(id, "env_id", None)
    if not isinstance(id, str):
        raise TypeError("Expected a registered environment instance, ID, EnvSpec, or task class with env_id")
    try:
        return _REGISTRY[id]
    except KeyError:
        raise KeyError(f"Unknown environment {id!r}; use list_envs() to discover available IDs") from None


def register_task(id: str, base: str, *, description: str, **overrides: Any) -> EnvSpec:
    """Create a task variant without copying an environment implementation."""
    spec = get_spec(base)
    return register_env(replace(spec, id=id, description=description, kwargs={**spec.kwargs, **overrides}))


def make(id: str | EnvSpec | type[BenchmarkEnv | LegacyEnvironment], **overrides: Any) -> BenchmarkEnv:
    """Instantiate only the selected backend. Constructor overrides win over defaults."""
    spec = get_spec(id)
    if isinstance(id, type) and issubclass(id, LegacyEnvironment):
        return id(**overrides).as_benchmark()
    factory = id if isinstance(id, type) else spec.entry_point
    if isinstance(factory, str):
        module, attr = factory.split(":", 1)
        factory = getattr(import_module(module), attr)
    settings = deepcopy({**spec.kwargs, **overrides})
    env = factory(**settings)
    if not hasattr(env, "settings"):
        env.settings = constructor_settings(factory, settings)
    env.spec = spec
    return env


def constructor_settings(factory, kwargs):
    """Record resolved constructor defaults as well as caller overrides."""
    bound = signature(factory).bind(**kwargs)
    bound.apply_defaults()
    return deepcopy(dict(bound.arguments))


@contextmanager
def environment_scope(environment, **kwargs):
    """Borrow existing instances; construct and close environments supplied as recipes."""
    if is_environment(environment):
        if kwargs:
            raise ValueError("Configure an existing world in its constructor; do not also pass env_kwargs")
        yield environment.as_benchmark() if isinstance(environment, LegacyEnvironment) else environment
    else:
        with make(environment, **kwargs) as env:
            yield env
