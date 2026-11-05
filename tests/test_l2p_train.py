"""Learning to Paint training pieces (T-029): the training subset helpers, the replay buffer, the environment and the DDPG update."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from brushstroke.baselines.l2p import trainset as ts
from brushstroke.baselines.l2p.actor import Actor, load_actor
from brushstroke.baselines.l2p.ddpg import (
    DDPG,
    Critic,
    DDPGConfig,
    Discriminator,
    PaintEnv,
    Replay,
    hard_update,
    soft_update,
    stage,
    train_agent,
)
from brushstroke.baselines.l2p.renderer import FCN


def png(w: int, h: int, noise: bool = True) -> bytes:
    rng = np.random.default_rng(0)
    arr = (
        rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        if noise
        else np.full((h, w, 3), 90, np.uint8)
    )
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def test_stratum_and_artist_rules() -> None:
    assert ts.stratum("Paintings", "European Paintings") == "painting"
    assert ts.stratum("Photographs|Prints", "Photographs") == "photograph"
    assert ts.stratum("Prints", "Drawings and Prints") == "print"
    assert ts.stratum("Vases", "Greek and Roman Art") == "object"
    assert ts.artist_names("Vincent van Gogh|  Unknown ") == ["vincent van gogh"]
    assert ts.artist_names("") == []


def test_plan_excludes_eval_objects_and_artists_and_is_seeded() -> None:
    cands = [
        {
            "id": i,
            "stratum": "painting" if i % 2 else "print",
            "artists": [f"a{i % 5}"],
            "number": "",
        }
        for i in range(1, 41)
    ]
    cfg = ts.TrainsetConfig()
    order = ts.plan(cands, cfg, {7, 8}, {"a3"})
    ids = [c["id"] for c in order]
    assert 7 not in ids and 8 not in ids and all(i % 5 != 3 for i in ids)
    strata = [c["stratum"] for c in order]
    assert strata == sorted(strata, key=ts.STRATA.index)
    assert ids == [c["id"] for c in ts.plan(cands, cfg, {7, 8}, {"a3"})]


def test_exclusions_read_the_real_eval_manifest() -> None:
    ids, artists = ts.exclusions(Path("configs/eval/eval_set.yaml"))
    assert len(ids) >= 40 and all(isinstance(i, int) for i in ids) and artists


def test_crop_resize_rejections_and_output() -> None:
    arr, why = ts.crop_resize(png(300, 280))
    assert arr is not None and arr.shape == (128, 128, 3) and arr.dtype == np.uint8 and why == ""
    assert ts.crop_resize(png(200, 200))[1] == "small"
    assert ts.crop_resize(png(900, 300))[1] == "aspect"
    assert ts.crop_resize(png(300, 300, noise=False))[1] == "flat"
    assert ts.crop_resize(b"not an image")[1] == "undecodable"


def test_near_duplicates_catch_mirrors_and_ignore_unrelated() -> None:
    rng = np.random.default_rng(1)
    a = np.asarray(
        Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).filter(
            ImageFilter.GaussianBlur(6)
        )
    )
    b = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    ref = {"eval": (ts.dhash(a), ts.dhash(a[:, ::-1].copy()))}
    found = ts.near_duplicates({1: ts.dhash(a[:, ::-1].copy()), 2: ts.dhash(b)}, ref, 6)
    assert found == {1: "eval"}


def test_manifest_digest_is_order_independent() -> None:
    assert ts.manifest_digest([(2, "b"), (1, "a")]) == ts.manifest_digest([(1, "a"), (2, "b")])
    assert ts.manifest_digest([(1, "a")]) != ts.manifest_digest([(1, "c")])


def test_networks_have_the_official_shapes_and_weight_norm_keys() -> None:
    critic, disc = Critic(), Discriminator()
    assert critic(torch.zeros(2, 12, 128, 128)).shape == (2, 1)
    assert disc(torch.zeros(2, 6, 128, 128)).shape == (2, 1)
    assert "conv1.weight_g" in critic.state_dict()
    assert "layer1.0.relu_1.alpha" in critic.state_dict()
    assert not any("bn" in k for k in critic.state_dict())


def test_hard_update_shares_buffers_and_soft_update_moves_parameters() -> None:
    a, b = Actor(), Actor()
    hard_update(b, a)
    assert b.bn1.running_mean is a.bn1.running_mean
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
    with torch.no_grad():
        a.fc.weight.add_(1.0)
    before = b.fc.weight.clone()
    soft_update(b, a, 0.5)
    assert torch.allclose(b.fc.weight, before + 0.5)


def test_replay_returns_consistent_transitions_across_the_ring() -> None:
    T, B = 4, 3
    rep = Replay(transitions=8 * B, env_batch=B, max_step=T, device=torch.device("cpu"))
    for g in range(23):  # more than two ring turns
        ep = g // T
        canvas = torch.full((B, 3, 128, 128), g % 250, dtype=torch.uint8)
        gt = torch.full((B, 3, 128, 128), 100 + ep, dtype=torch.uint8)
        rep.add(canvas, torch.full((B, 65), float(g)), gt)
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        c0, c1, gt, t, action, done = rep.sample(32, gen)
        g = action[:, 0].long()  # the action stored the global step
        assert torch.equal(c1[:, 0, 0, 0].long(), g % 250)
        prev = torch.where(t == 0, torch.zeros_like(g), (g - 1) % 250)
        assert torch.equal(c0[:, 0, 0, 0].long(), prev)
        assert torch.equal(gt[:, 0, 0, 0].long(), 100 + g // T)
        assert torch.equal(done, t == T - 1)
        assert int(g.min()) >= 23 - rep.slots + 1


def test_env_quantizes_the_canvas_flips_and_orders_channels() -> None:
    imgs = np.zeros((4, 128, 128, 3), np.uint8)
    imgs[:, :, :, 0] = 200  # red in RGB
    imgs[:, :, :64, 1] = 50
    env = PaintEnv(imgs, 4, 2, torch.device("cpu"), bgr=True)
    env.reset(np.array([0, 1, 2, 3]), np.array([True, False, False, False]))
    assert env.gt.dtype == torch.uint8
    assert int(env.gt[1, 2, 5, 5]) == 200 and int(env.gt[1, 0, 5, 5]) == 0  # BGR
    assert int(env.gt[0, 1, 0, 0]) == 0 and int(env.gt[0, 1, 0, 127]) == 50  # flipped
    env.step(torch.rand(4, 65), FCN().eval())
    assert env.canvas.dtype == torch.uint8 and env.t == 1 and env.dist().shape == (4,)


def test_lr_stage_boundaries() -> None:
    cfg = DDPGConfig()
    assert [stage(e, cfg) for e in (1, 9999, 10000, 19999, 20000, 50000)] == [0, 0, 1, 1, 2, 2]


def tiny_setup(tmp: Path) -> DDPGConfig:
    rng = np.random.default_rng(0)
    np.save(tmp / "images.npy", rng.integers(0, 255, (12, 128, 128, 3), dtype=np.uint8))
    val = np.zeros(12, bool)
    val[:4] = True
    np.save(tmp / "is_val.npy", val)
    torch.manual_seed(0)
    torch.save(FCN().state_dict(), tmp / "renderer.pt")
    return DDPGConfig(
        episodes=3,
        max_step=2,
        env_batch=4,
        batch_size=4,
        rmsize=4,
        warmup=4,
        episode_train_times=2,
        trainset=str(tmp),
        renderer=str(tmp / "renderer.pt"),
        output_dir=str(tmp / "run"),
        device="cpu",
        validate_every=2,
        validate_images=4,
        log_every=1,
        checkpoint_every=1,
    )


@pytest.mark.slow
def test_training_runs_logs_saves_a_loadable_actor_and_resumes(tmp_path: Path) -> None:
    cfg = tiny_setup(tmp_path)
    actor_path = train_agent(cfg, log=False)
    lines = (tmp_path / "run" / "history.jsonl").read_text().splitlines()
    rows = [json.loads(x) for x in lines]
    assert [r["episode"] for r in rows] == [1, 2, 3] and "val_l2" in rows[1] and "q" in rows[2]
    assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, float))
    assert (tmp_path / "run" / "final_validation.json").exists()
    net = load_actor(actor_path)  # strict load of the saved state dict
    assert net(torch.zeros(1, 9, 128, 128)).shape == (1, 65)
    more = DDPGConfig(**{**cfg.__dict__, "episodes": 4})
    train_agent(more, log=False)
    last = (tmp_path / "run" / "history.jsonl").read_text().splitlines()[-1]
    assert json.loads(last)["episode"] == 4


@pytest.mark.slow
def test_update_changes_the_actor_and_keeps_targets_slow(tmp_path: Path) -> None:
    cfg = tiny_setup(tmp_path)
    dev = torch.device("cpu")
    agent = DDPG(cfg, FCN().eval(), dev)
    rep = Replay(16, 4, 2, dev)
    for _ in range(4):
        c = torch.randint(0, 255, (4, 3, 128, 128), dtype=torch.uint8)
        g = torch.randint(0, 255, (4, 3, 128, 128), dtype=torch.uint8)
        rep.add(c, torch.rand(4, 65), g)
    before = agent.actor.fc.weight.clone()
    tgt = agent.actor_target.fc.weight.clone()
    out = agent.update(rep, torch.Generator().manual_seed(0), 3e-4, 1e-3)
    assert all(np.isfinite(v) for v in out.values())
    assert not torch.equal(agent.actor.fc.weight, before)
    assert float((agent.actor_target.fc.weight - tgt).abs().max()) < 1e-2  # tau = 0.001
