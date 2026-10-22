"""Typed YAML configuration with CLI dot-overrides.

A config schema is a plain ``@dataclass``. :func:`load_config` merges, in order, the schema
defaults, the YAML file and the ``key.sub=value`` overrides, validates types against the schema
and returns an instance of the dataclass. :func:`dump_config` writes the fully resolved config
next to the outputs so that every run can be reproduced.

OmegaConf was picked over pydantic because it provides dot-list overrides, YAML I/O and
dataclass-based type validation in a single dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")

RESOLVED_CONFIG_NAME = "config.resolved.yaml"


def load_config(
    schema: type[T],
    path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> T:
    """Load a typed config.

    Args:
        schema: dataclass type giving the fields, types and defaults.
        path: optional YAML file; keys present override the schema defaults.
        overrides: dot-list overrides such as ``["optim.lr=0.01", "seed=3"]``; values are parsed
            as YAML.

    Returns:
        An instance of ``schema``.

    Raises:
        omegaconf.errors.ConfigKeyError: unknown key (in the YAML file or in an override).
        omegaconf.errors.ValidationError: value of the wrong type.
    """
    cfg = cast(DictConfig, OmegaConf.structured(schema))
    if path is not None:
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.load(Path(path))))
    if overrides:
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides))))
    return cast(T, OmegaConf.to_object(cfg))


def dump_config(cfg: object, out_dir: str | Path, name: str = RESOLVED_CONFIG_NAME) -> Path:
    """Write the resolved config (a dataclass instance) as YAML in ``out_dir``; return its path."""
    if not is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError(f"expected a dataclass instance, got {type(cfg).__name__}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / name
    data: dict[str, Any] = asdict(cfg)
    target.write_text(OmegaConf.to_yaml(OmegaConf.create(data)), encoding="utf-8")
    return target
