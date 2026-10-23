"""Progressive coarse-to-fine optimization of strokes through the differentiable renderer.

Algorithm (after Stylized Neural Painting, Zou et al., CVPR 2021, with our analytic renderer):

1. The canvas starts as a flat background colour.
2. For each level ``g`` of ``levels`` (grid ``g x g``): ``strokes_per_patch`` new strokes per patch
   are initialized from the residual error (:mod:`.init`), then *all new strokes of the level are
   optimized jointly* for ``iters`` steps (RMSprop/Adam on sigmoid-reparameterized parameters)
   against the loss (:mod:`.losses`), rendered over the canvas of the previous levels, which is
   frozen unless ``finetune_all``.
3. The level's strokes are appended in order (patch-major) and the canvas is updated.

The output is a :class:`~brushstroke.strokes.StrokeSequence` ordered level by level with
``scale_level`` (index in ``levels``) and ``patch_id`` (patch index inside the level) tags.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable

import torch

from brushstroke.baselines.optim.config import LossConfig, OptimConfig
from brushstroke.baselines.optim.init import cosine_lr, dead_strokes, sample_strokes
from brushstroke.baselines.optim.losses import CombinedLoss
from brushstroke.renderer.brushes import BrushBank, BrushConfig, get_brush_bank
from brushstroke.renderer.params import params_from_unconstrained, unconstrained_from_params
from brushstroke.renderer.pipeline import RenderConfig, render
from brushstroke.strokes import BrushSpec, StrokeSequence
from brushstroke.utils.device import peak_vram_mb, reset_peak_vram

BYTES_PER_STROKE_PIXEL = 44  # measured activation memory of forward+backward (see renderer.md)
LevelCallback = Callable[[int, torch.Tensor, StrokeSequence], None]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _make_optimizer(name: str, params: list[dict[str, object]], lr: float) -> torch.optim.Optimizer:
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unknown optimizer {name!r}; expected 'rmsprop' or 'adam'")


def _reset_state_rows(opt: torch.optim.Optimizer, param: torch.Tensor, rows: torch.Tensor) -> None:
    """Zero the optimizer statistics of the given stroke rows (after re-initializing them)."""
    for value in opt.state.get(param, {}).values():
        if isinstance(value, torch.Tensor) and value.shape == param.shape:
            value[:, rows] = 0


def config_hash(*cfgs: object) -> str:
    """Short stable hash of dataclass configs (stored in the sequence meta)."""
    blob = json.dumps([dataclasses.asdict(c) for c in cfgs], sort_keys=True)  # type: ignore[call-overload]
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def paint_optim(
    target: torch.Tensor,
    optim_cfg: OptimConfig,
    loss_cfg: LossConfig | None = None,
    brush_cfg: BrushConfig | None = None,
    render_cfg: RenderConfig | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    callback: LevelCallback | None = None,
) -> StrokeSequence:
    """Decompose ``target`` into ordered strokes by progressive optimization.

    Args:
        target: ``(3, H, W)`` image in [0, 1]; ``H, W`` must equal ``optim_cfg.height/width``.
        optim_cfg: levels, strokes per patch, steps, optimizer, initialization, flags.
        loss_cfg: loss weights (default: pixel L1 only).
        brush_cfg: brush kind / bank (default ``BrushConfig()``).
        render_cfg: chunking etc. (``use_checkpoint`` is chosen per level from the memory estimate).
        seed: seed of every random draw (initialization, brush variants).
        device: torch device of the optimization.
        callback: called after each level as ``callback(level_index, canvas (3, H, W), sequence)``.

    Returns:
        A :class:`StrokeSequence` of canvas size ``(H, W)``; ``meta`` holds the config hash, seed,
        per-level losses and timings, total runtime and peak VRAM.
    """
    cfg = optim_cfg
    loss_cfg = loss_cfg or LossConfig()
    brush_cfg = brush_cfg or BrushConfig()
    render_cfg = render_cfg or RenderConfig()
    dev = torch.device(device)
    h, w = cfg.height, cfg.width
    if target.shape != (3, h, w):
        raise ValueError(f"target must have shape (3, {h}, {w}), got {tuple(target.shape)}")
    if len(cfg.levels) != len(cfg.strokes_per_patch) or not cfg.levels:
        raise ValueError("levels and strokes_per_patch must be non-empty and of equal length")
    if cfg.iters < 1 or any(i < 1 for i in cfg.iters_per_level):
        raise ValueError("iters (and iters_per_level) must be >= 1")
    if cfg.iters_per_level and len(cfg.iters_per_level) != len(cfg.levels):
        raise ValueError("iters_per_level must be empty or have one entry per level")

    spec = BrushSpec(
        kind=brush_cfg.kind,
        bank_size=brush_cfg.bank_size,
        mask_height=brush_cfg.mask_height,
        mask_width=brush_cfg.mask_width,
        seed=brush_cfg.seed,
        index_seed=render_cfg.brush_seed,
    )
    bank = get_brush_bank(spec.kind, spec.bank_size, (spec.mask_height, spec.mask_width), spec.seed)
    bank = bank.to(dev)
    loss_fn = CombinedLoss(loss_cfg)
    gen = torch.Generator().manual_seed(seed)
    tgt = target.to(dev, torch.float32)
    tgt_b = tgt[None]
    background = torch.tensor(cfg.background, dtype=torch.float32, device=dev).view(3, 1, 1)
    canvas = background.expand(3, h, w).contiguous()

    params = torch.zeros(0, 9)
    brush_idx = torch.zeros(0, dtype=torch.long)
    level_tags = torch.zeros(0, dtype=torch.int32)
    patch_tags = torch.zeros(0, dtype=torch.int32)
    curves: list[list[float]] = []
    level_seconds: list[float] = []
    level_final: list[dict[str, float]] = []
    reset_peak_vram()
    _sync(dev)
    t_start = time.perf_counter()

    for li, (grid, per_patch) in enumerate(zip(cfg.levels, cfg.strokes_per_patch, strict=True)):
        t_level = time.perf_counter()
        patch_ids = torch.arange(grid * grid).repeat_interleave(per_patch)
        m = patch_ids.numel()
        init = sample_strokes(tgt, canvas, grid, patch_ids, cfg, gen, per_patch)
        new_idx = torch.randint(0, spec.bank_size, (m,), generator=gen)
        raw = unconstrained_from_params(init[None]).to(dev).requires_grad_()
        groups: list[dict[str, object]] = [{"params": [raw], "lr": cfg.lr}]
        prev_raw: torch.Tensor | None = None
        if cfg.finetune_all and params.shape[0] > 0:
            prev_raw = unconstrained_from_params(params[None]).to(dev).requires_grad_()
            groups.append({"params": [prev_raw], "lr": cfg.lr * cfg.finetune_lr_factor})
        opt = _make_optimizer(cfg.optimizer, groups, cfg.lr)
        base_lrs = [float(g["lr"]) for g in groups]  # type: ignore[arg-type]

        total = m + (params.shape[0] if prev_raw is not None else 0)
        use_ckpt = total * h * w * BYTES_PER_STROKE_PIXEL > cfg.checkpoint_bytes
        level_render = dataclasses.replace(
            render_cfg, use_checkpoint=use_ckpt or render_cfg.use_checkpoint
        )
        idx_new = new_idx.to(dev)[None]
        idx_prev = brush_idx.to(dev)[None]
        losses: list[torch.Tensor] = []

        n_iters = cfg.iters_per_level[li] if cfg.iters_per_level else cfg.iters
        for it in range(n_iters):
            for g, base in zip(opt.param_groups, base_lrs, strict=True):
                g["lr"] = cosine_lr(base, cfg.lr_final_frac, it, n_iters)
            opt.zero_grad(set_to_none=True)
            new_params = params_from_unconstrained(raw)
            if prev_raw is not None:
                allp = torch.cat([params_from_unconstrained(prev_raw), new_params], dim=1)
                img = render(
                    allp,
                    bank,
                    h,
                    w,
                    canvas=background.view(3),
                    cfg=level_render,
                    brush_index=torch.cat([idx_prev, idx_new], dim=1),
                )[0]
            else:
                img = render(
                    new_params, bank, h, w, canvas=canvas, cfg=level_render, brush_index=idx_new
                )[0]
            loss = loss_fn(img[None], tgt_b)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            losses.append(loss.detach())
            if cfg.resample_every > 0 and (it + 1) % cfg.resample_every == 0 and it + 1 < n_iters:
                with torch.no_grad():
                    cur = params_from_unconstrained(raw)[0].cpu()
                    dead = dead_strokes(cur, cfg)
                    if bool(dead.any()):
                        rows = torch.nonzero(dead).flatten()
                        fresh = sample_strokes(
                            tgt, img.detach(), grid, patch_ids[rows], cfg, gen, per_patch
                        )
                        raw[0, rows] = unconstrained_from_params(fresh[None])[0].to(dev)
                        _reset_state_rows(opt, raw, rows.to(dev))

        with torch.no_grad():
            final_new = params_from_unconstrained(raw)[0].cpu()
            if prev_raw is not None:
                params = params_from_unconstrained(prev_raw)[0].cpu()
            params = torch.cat([params, final_new])
            brush_idx = torch.cat([brush_idx, new_idx])
            level_tags = torch.cat([level_tags, torch.full((m,), li, dtype=torch.int32)])
            patch_tags = torch.cat([patch_tags, patch_ids.to(torch.int32)])
            canvas = (
                render(
                    params[None].to(dev),
                    bank,
                    h,
                    w,
                    canvas=background.view(3),
                    cfg=level_render,
                    brush_index=brush_idx[None].to(dev),
                )[0]
                if prev_raw is not None
                else render(
                    final_new[None].to(dev),
                    bank,
                    h,
                    w,
                    canvas=canvas,
                    cfg=level_render,
                    brush_index=idx_new,
                )[0]
            )
        curve = [float(v) for v in torch.stack(losses).cpu()]
        curves.append(curve)
        _sync(dev)
        level_seconds.append(time.perf_counter() - t_level)
        with torch.no_grad():
            level_final.append(
                {k: float(v) for k, v in loss_fn.terms(canvas[None], tgt_b).items()}
                | {"l1_metric": float((canvas - tgt).abs().mean())}
            )
        if callback is not None:
            callback(
                li, canvas, _build(params, spec, h, w, brush_idx, level_tags, patch_tags, cfg, {})
            )

    final_curve: list[float] = []
    final_seconds = 0.0
    if cfg.final_pass_iters > 0 and params.shape[0] > 0:
        t_final = time.perf_counter()
        canvas, params, final_curve = _final_pass(
            params, brush_idx, bank, background, tgt, loss_fn, cfg, render_cfg, dev
        )
        _sync(dev)
        final_seconds = time.perf_counter() - t_final
    with torch.no_grad():
        final_loss = {k: float(v) for k, v in loss_fn.terms(canvas[None], tgt_b).items()} | {
            "l1_metric": float((canvas - tgt).abs().mean())
        }
    _sync(dev)
    meta = {
        "producer": "optim_baseline",
        "seed": seed,
        "config_hash": config_hash(cfg, loss_cfg, brush_cfg, render_cfg),
        "levels": list(cfg.levels),
        "strokes_per_patch": list(cfg.strokes_per_patch),
        "loss_curve": curves,
        "level_final_loss": level_final,
        "level_seconds": level_seconds,
        "final_pass_curve": final_curve,
        "final_pass_seconds": final_seconds,
        "final_loss": final_loss,
        "runtime_seconds": time.perf_counter() - t_start,
        "peak_vram_mb": peak_vram_mb(),
        "torch_version": torch.__version__,
        "device": str(dev),
    }
    return _build(params, spec, h, w, brush_idx, level_tags, patch_tags, cfg, meta)


def _final_pass(
    params: torch.Tensor,
    brush_idx: torch.Tensor,
    bank: BrushBank,
    background: torch.Tensor,
    tgt: torch.Tensor,
    loss_fn: CombinedLoss,
    cfg: OptimConfig,
    render_cfg: RenderConfig,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Jointly fine-tune all strokes (``final_pass_iters`` steps): canvas, params, losses."""
    h, w = cfg.height, cfg.width
    n = params.shape[0]
    base_lr = cfg.lr * cfg.final_pass_lr_factor
    raw = unconstrained_from_params(params[None]).to(dev).requires_grad_()
    opt = _make_optimizer(cfg.optimizer, [{"params": [raw], "lr": base_lr}], base_lr)
    use_ckpt = n * h * w * BYTES_PER_STROKE_PIXEL > cfg.checkpoint_bytes
    rcfg = dataclasses.replace(render_cfg, use_checkpoint=use_ckpt or render_cfg.use_checkpoint)
    idx = brush_idx.to(dev)[None]
    losses: list[torch.Tensor] = []
    for it in range(cfg.final_pass_iters):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(base_lr, cfg.lr_final_frac, it, cfg.final_pass_iters)
        opt.zero_grad(set_to_none=True)
        img = render(
            params_from_unconstrained(raw),
            bank,
            h,
            w,
            canvas=background.view(3),
            cfg=rcfg,
            brush_index=idx,
        )[0]
        loss = loss_fn(img[None], tgt[None])
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        losses.append(loss.detach())
    with torch.no_grad():
        new_params = params_from_unconstrained(raw)[0]
        canvas = render(
            new_params[None], bank, h, w, canvas=background.view(3), cfg=rcfg, brush_index=idx
        )[0]
    return canvas, new_params.cpu(), [float(v) for v in torch.stack(losses).cpu()]


def _build(
    params: torch.Tensor,
    spec: BrushSpec,
    h: int,
    w: int,
    brush_idx: torch.Tensor,
    level_tags: torch.Tensor,
    patch_tags: torch.Tensor,
    cfg: OptimConfig,
    meta: dict[str, object],
) -> StrokeSequence:
    return StrokeSequence(
        params=params,
        canvas_size=(h, w),
        brush=spec,
        brush_index=brush_idx,
        background=(cfg.background[0], cfg.background[1], cfg.background[2]),
        scale_level=level_tags,
        patch_id=patch_tags,
        source=["optim_baseline"] * params.shape[0],
        meta=meta,
    )
