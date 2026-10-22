"""Canonical stroke-sequence data model: an ordered list of strokes plus metadata.

Every producer (optimization baseline, Paint Transformer, RL agent, contribution modules) emits a
:class:`StrokeSequence` and every consumer (animation, evaluation) reads one. It lives at the top
level of the package (not in ``renderer/``) because it is shared by all subpackages and is not
renderer-specific; rendering is delegated to :mod:`brushstroke.renderer`.

Conventions
-----------
* ``params`` is ``(N, 9)`` float32 in the renderer convention (see
  :mod:`brushstroke.renderer.params`): ``x, y, w, h, theta, r, g, b, a``, all normalized, so a
  sequence can be re-rendered at any resolution.
* Strokes are ordered: stroke ``k`` is painted over strokes ``0..k-1``.
* Per-stroke tags are arrays of length N: ``brush_index`` (variant in the brush bank),
  ``scale_level`` and ``patch_id`` (``-1`` = unknown) and ``source`` (producer name, ``""`` =
  unset).
* ``meta`` is a free-form JSON-serializable dict (config hash, seed, producer version, timings...).
  It is normalized through a JSON round trip at construction, so tuples become lists, dict keys
  become strings (``{1: 'x'}`` -> ``{'1': 'x'}``) and save/load is exactly the identity.
* On disk: a single ``.npz`` holding the float32/int arrays plus a UTF-8 JSON ``header``.
  ``schema_version`` (see ``docs/design/stroke_sequence.md``) is checked on load.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brushstroke.renderer.brushes import BrushBank, get_brush_bank
from brushstroke.renderer.params import NUM_PARAMS
from brushstroke.renderer.pipeline import RenderConfig, render, render_progressive
from brushstroke.renderer.rasterize import default_brush_index

SCHEMA_VERSION = 1
_BANK_CACHE_SIZE = 8
HEADER_KEY = "header"
UNSET = -1

# schema_version -> function upgrading (header, arrays) from that version to the next one.
_MIGRATIONS: dict[int, Callable[[dict[str, Any], dict[str, np.ndarray]], None]] = {}


@dataclass(frozen=True)
class BrushSpec:
    """Which brush bank the strokes were made for (the masks are regenerated from this spec).

    ``index_seed`` must equal the ``RenderConfig.brush_seed`` the producer rendered with if it
    relied on the renderer's default brush draw (see :meth:`StrokeSequence.from_batch`).
    """

    kind: str = "oil"
    bank_size: int = 8
    mask_height: int = 64
    mask_width: int = 128
    seed: int = 0  # seed of the brush masks
    index_seed: int = 0  # seed of the default per-stroke variant draw (RenderConfig.brush_seed)


@lru_cache(maxsize=_BANK_CACHE_SIZE)
def _device_bank(spec: BrushSpec, device: str) -> BrushBank:
    """Brush bank of ``spec`` on ``device`` (cached: prefix renders reuse it)."""
    bank = get_brush_bank(spec.kind, spec.bank_size, (spec.mask_height, spec.mask_width), spec.seed)
    return bank.to(device)


def _canonical_meta(meta: dict[str, Any]) -> dict[str, Any]:
    try:
        out = json.loads(json.dumps(meta, allow_nan=True))
    except TypeError as exc:
        raise TypeError(f"StrokeSequence.meta must be JSON-serializable: {exc}") from exc
    if not isinstance(out, dict):
        raise TypeError("StrokeSequence.meta must be a dict")
    return out


@dataclass
class StrokeSequence:
    """An ordered list of N strokes and the metadata needed to render and interpret it.

    Attributes:
        params: ``(N, 9)`` stroke parameters (converted to float32).
        canvas_size: default render size ``(H, W)`` in pixels.
        brush: :class:`BrushSpec` of the bank the strokes use.
        brush_index: ``(N,)`` long tensor, bank variant of each stroke. ``None`` = painting 0 of
            ``default_brush_index(1, N, bank_size, brush.index_seed)``, i.e. exactly what the
            renderer draws for a single painting rendered with ``brush_seed=index_seed``. For
            painting ``b > 0`` of a batch use :meth:`from_batch`.
        background: initial canvas colour ``(r, g, b)`` in [0, 1].
        background_ref: optional free-text reference to a background image (name/path/hash);
            informational only, the renderer uses ``background``.
        scale_level: ``(N,)`` int32 tag, e.g. the coarse-to-fine level of the producer (-1 = unset).
        patch_id: ``(N,)`` int32 tag, e.g. the patch the stroke was predicted for (-1 = unset).
        source: length-N list of producer names (``""`` = unset).
        meta: free-form JSON-serializable metadata.
        schema_version: version of the data model (see :data:`SCHEMA_VERSION`).
    """

    params: torch.Tensor
    canvas_size: tuple[int, int] = (512, 512)
    brush: BrushSpec = field(default_factory=BrushSpec)
    brush_index: torch.Tensor | None = None
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    background_ref: str | None = None
    scale_level: torch.Tensor | None = None
    patch_id: torch.Tensor | None = None
    source: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.params.ndim != 2 or self.params.shape[1] != NUM_PARAMS:
            raise ValueError(
                f"params must have shape (N, {NUM_PARAMS}), got {tuple(self.params.shape)}"
            )
        self.params = self.params.detach().to(torch.float32).contiguous()
        n, device = self.params.shape[0], self.params.device
        height, width = self.canvas_size
        if height < 1 or width < 1:
            raise ValueError(f"canvas_size must be positive, got {self.canvas_size}")
        self.canvas_size = (int(height), int(width))
        self.background = tuple(float(c) for c in self.background)  # type: ignore[assignment]
        if len(self.background) != 3:
            raise ValueError(f"background must be an (r, g, b) triple, got {self.background}")
        if self.brush_index is None:
            self.brush_index = default_brush_index(
                1, n, self.brush.bank_size, self.brush.index_seed
            )[0]
        self.brush_index = self._tag(self.brush_index, torch.int64, "brush_index", device)
        if n and not (
            int(self.brush_index.min()) >= 0 and int(self.brush_index.max()) < self.brush.bank_size
        ):
            raise ValueError(f"brush_index must lie in [0, {self.brush.bank_size})")
        self.scale_level = self._tag(self.scale_level, torch.int32, "scale_level", device)
        self.patch_id = self._tag(self.patch_id, torch.int32, "patch_id", device)
        self.source = [""] * n if self.source is None else [str(s) for s in self.source]
        if len(self.source) != n:
            raise ValueError(f"source must have length {n}, got {len(self.source)}")
        self.meta = _canonical_meta(self.meta)

    def _tag(
        self,
        value: torch.Tensor | Sequence[int] | None,
        dtype: torch.dtype,
        name: str,
        device: torch.device,
    ) -> torch.Tensor:
        n = self.params.shape[0]
        if value is None:
            return torch.full((n,), UNSET, dtype=dtype, device=device)
        t = torch.as_tensor(value).detach().to(device=device, dtype=dtype).contiguous()
        if t.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {tuple(t.shape)}")
        return t

    # ------------------------------------------------------------------ construction

    @classmethod
    def empty(cls, canvas_size: tuple[int, int] = (512, 512), **kwargs: Any) -> StrokeSequence:
        """A sequence with no stroke (kwargs as the constructor, except ``params``)."""
        return cls(params=torch.zeros(0, NUM_PARAMS), canvas_size=canvas_size, **kwargs)

    @classmethod
    def from_batch(
        cls, params: torch.Tensor, b: int = 0, brush: BrushSpec | None = None, **kwargs: Any
    ) -> StrokeSequence:
        """Sequence of painting ``b`` of a producer's batch ``(B, N, 9)``.

        Uses the brush variants the renderer draws by default for that painting:
        ``default_brush_index(B, N, bank_size, brush.index_seed)[b]`` (independent of ``B``). So
        ``from_batch(p, b, spec).render()`` equals the producer's
        ``render(p, ..., cfg=RenderConfig(brush_seed=spec.index_seed))[b]``. If the producer passed
        an explicit ``brush_index`` to the renderer, give ``brush_index=`` here (kwargs are
        forwarded to the constructor).
        """
        if params.ndim != 3:
            raise ValueError(f"expected params of shape (B, N, 9), got {tuple(params.shape)}")
        spec = brush or BrushSpec()
        if "brush_index" not in kwargs:
            bsz, n = params.shape[:2]
            kwargs["brush_index"] = default_brush_index(bsz, n, spec.bank_size, spec.index_seed)[b]
        return cls(params=params[b], brush=spec, **kwargs)

    @classmethod
    def concat(cls, parts: Sequence[StrokeSequence]) -> StrokeSequence:
        """Concatenate sequences in order (strokes of ``parts[1]`` are painted over ``parts[0]``).

        All parts must share ``canvas_size``, ``brush`` and ``background``. Tags and sources are
        concatenated; ``meta`` and ``background_ref`` are taken from the first part.
        """
        if not parts:
            raise ValueError("concat needs at least one sequence")
        first = parts[0]
        for p in parts[1:]:
            if (p.canvas_size, p.brush, p.background) != (
                first.canvas_size,
                first.brush,
                first.background,
            ):
                raise ValueError(
                    "cannot concat sequences with different canvas_size, brush or background"
                )
        device = first.params.device

        def cat(name: str) -> torch.Tensor:
            return torch.cat([getattr(p, name).to(device) for p in parts])

        return cls(
            params=cat("params"),
            canvas_size=first.canvas_size,
            brush=first.brush,
            brush_index=cat("brush_index"),
            background=first.background,
            background_ref=first.background_ref,
            scale_level=cat("scale_level"),
            patch_id=cat("patch_id"),
            source=[s for p in parts for s in p.source or []],
            meta=first.meta,
            schema_version=first.schema_version,
        )

    # ------------------------------------------------------------------ basic operations

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def select(self, index: slice | Sequence[int] | torch.Tensor) -> StrokeSequence:
        """New sequence with the strokes at ``index`` (slice, int list/tensor or bool mask)."""
        n = len(self)
        if isinstance(index, slice):
            ids = torch.arange(n, device=self.params.device)[index]
        else:
            ids = torch.as_tensor(index, device=self.params.device)
            if ids.dtype == torch.bool:
                if ids.shape != (n,):
                    raise ValueError(f"boolean mask must have shape ({n},), got {tuple(ids.shape)}")
                ids = torch.nonzero(ids).flatten()
            elif ids.numel() and (int(ids.min()) < -n or int(ids.max()) >= n):
                raise IndexError(f"stroke index out of range for a sequence of {n} strokes")
        ids = ids.long()
        assert self.brush_index is not None and self.scale_level is not None
        assert self.patch_id is not None and self.source is not None
        return StrokeSequence(
            params=self.params[ids],
            canvas_size=self.canvas_size,
            brush=self.brush,
            brush_index=self.brush_index[ids],
            background=self.background,
            background_ref=self.background_ref,
            scale_level=self.scale_level[ids],
            patch_id=self.patch_id[ids],
            source=[self.source[i] for i in ids.tolist()],
            meta=self.meta,
            schema_version=self.schema_version,
        )

    def __getitem__(self, index: slice | Sequence[int] | torch.Tensor) -> StrokeSequence:
        """``seq[:k]`` is the prefix of the first k strokes (tags and meta are kept)."""
        return self.select(index)

    def reorder(self, perm: Sequence[int] | torch.Tensor) -> StrokeSequence:
        """New sequence whose k-th stroke is ``self[perm[k]]`` (params and all tags permuted).

        Args:
            perm: a permutation of ``range(N)``; anything else raises ``ValueError``.
        """
        p = torch.as_tensor(perm, device=self.params.device)
        if p.ndim != 1 or p.numel() != len(self) or p.dtype == torch.bool or p.is_floating_point():
            raise ValueError(f"perm must be an integer vector of length {len(self)}")
        if not torch.equal(torch.sort(p).values, torch.arange(len(self), device=p.device)):
            raise ValueError("perm is not a permutation of range(N)")
        return self.select(p)

    def to(self, device: torch.device | str) -> StrokeSequence:
        """Copy with all tensors (params and tags) on ``device``; ``source``/``meta`` unchanged."""
        out = self.select(slice(None))
        for name in ("params", "brush_index", "scale_level", "patch_id"):
            setattr(out, name, getattr(out, name).to(device))
        return out

    # ------------------------------------------------------------------ rendering

    def render(
        self,
        height: int | None = None,
        width: int | None = None,
        cfg: RenderConfig | None = None,
        canvas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render the sequence: ``(3, H, W)`` float32 in [0, 1] on the device of ``params``.

        Args:
            height, width: output size; both default to ``canvas_size`` (give both or none).
                Parameters are normalized, so any resolution is valid.
            cfg: render config (chunking, checkpointing, ...).
            canvas: optional initial canvas ``(3, H, W)`` replacing ``background``.
        """
        if (height is None) != (width is None):
            raise ValueError("give both height and width, or neither")
        h, w = (height, width) if height is not None and width is not None else self.canvas_size
        assert self.brush_index is not None
        out = render(
            self.params[None],
            _device_bank(self.brush, str(self.params.device)),
            h,
            w,
            canvas=self.background if canvas is None else canvas,
            cfg=cfg,
            brush_index=self.brush_index[None],
        )
        return out[0]

    def render_progressive(
        self,
        checkpoints: Sequence[int],
        height: int | None = None,
        width: int | None = None,
        cfg: RenderConfig | None = None,
        canvas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Canvases after the first ``k`` strokes for each ``k`` in ``checkpoints``, in one pass.

        Returns ``(len(checkpoints), 3, H, W)``; entry ``i`` is ``self[:checkpoints[i]].render()``.
        Arguments as :meth:`render`; checkpoints lie in ``[0, N]``.
        """
        if (height is None) != (width is None):
            raise ValueError("give both height and width, or neither")
        h, w = (height, width) if height is not None and width is not None else self.canvas_size
        assert self.brush_index is not None
        out = render_progressive(
            self.params[None],
            _device_bank(self.brush, str(self.params.device)),
            h,
            w,
            checkpoints,
            canvas=self.background if canvas is None else canvas,
            cfg=cfg,
            brush_index=self.brush_index[None],
        )
        return out[0]

    # ------------------------------------------------------------------ serialization

    def _header(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canvas_size": list(self.canvas_size),
            "brush": asdict(self.brush),
            "background": list(self.background),
            "background_ref": self.background_ref,
            "source": self.source,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        """Write a single ``.npz`` (arrays + JSON header); parent directories are created.

        Returns the path actually written (``.npz`` is appended if missing).
        """
        out = Path(path)
        if out.suffix != ".npz":
            out = out.with_name(out.name + ".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        assert (
            self.brush_index is not None
            and self.scale_level is not None
            and self.patch_id is not None
        )
        header = json.dumps(self._header(), sort_keys=True, allow_nan=True).encode("utf-8")
        np.savez(
            out,
            params=self.params.cpu().numpy().astype(np.float32),
            brush_index=self.brush_index.cpu().numpy().astype(np.int64),
            scale_level=self.scale_level.cpu().numpy().astype(np.int32),
            patch_id=self.patch_id.cpu().numpy().astype(np.int32),
            header=np.frombuffer(header, dtype=np.uint8),
        )
        return out

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> StrokeSequence:
        """Read a file written by :meth:`save` (older schema versions are migrated).

        Raises:
            ValueError: not a stroke-sequence file, or its ``schema_version`` is too new.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            if HEADER_KEY not in data.files:
                raise ValueError(f"{path} is not a StrokeSequence file (no '{HEADER_KEY}' entry)")
            header: dict[str, Any] = json.loads(bytes(data[HEADER_KEY].tobytes()).decode("utf-8"))
            arrays = {k: data[k] for k in data.files if k != HEADER_KEY}
        version = int(header.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema_version {version}, newer than the supported {SCHEMA_VERSION}; "
                "upgrade brushstroke to read it"
            )
        while version < SCHEMA_VERSION:
            if version not in _MIGRATIONS:
                raise ValueError(f"{path}: no migration from schema_version {version}")
            _MIGRATIONS[version](header, arrays)
            version += 1
            header["schema_version"] = version
        h, w = header["canvas_size"]
        return cls(
            params=torch.from_numpy(arrays["params"]).to(device),
            canvas_size=(int(h), int(w)),
            brush=BrushSpec(**header["brush"]),
            brush_index=torch.from_numpy(arrays["brush_index"]).to(device),
            background=tuple(header["background"]),
            background_ref=header["background_ref"],
            scale_level=torch.from_numpy(arrays["scale_level"]).to(device),
            patch_id=torch.from_numpy(arrays["patch_id"]).to(device),
            source=header["source"],
            meta=header["meta"],
            schema_version=version,
        )

    def equals(self, other: StrokeSequence) -> bool:
        """Exact equality of params, tags, background, brush, canvas size, source and meta."""
        a, b = self.to("cpu"), other.to("cpu")
        tensors_equal = all(
            torch.equal(getattr(a, n), getattr(b, n))
            for n in ("params", "brush_index", "scale_level", "patch_id")
        )
        return tensors_equal and a._header() == b._header()
