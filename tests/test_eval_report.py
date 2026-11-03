"""Tests of the tables, the statistics families, the figure and the tau rule (T-026)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch

from brushstroke.eval import report as rp
from brushstroke.eval import runner as rn
from brushstroke.eval.protocol import load_protocol
from scripts import eval_run
from tests.test_eval_runner import SIZE, FakeProducer, context

PROTO = load_protocol()
pytestmark = pytest.mark.slow  # builds a 10-image run on the CPU (about 30 s)


class Converging(FakeProducer):
    """Moves linearly to the target over ``speed`` strokes (smaller = faster)."""

    def __init__(self, name: str, total: int, speed: int, dagger: tuple[str, ...] = ()) -> None:
        super().__init__(total)
        self.name, self.speed, self.dagger_metrics = name, speed, dagger
        self.replicate_kind = "init seed"

    def paint(self, target: torch.Tensor, checkpoints: Any, seed: int) -> Any:
        res = super().paint(target, checkpoints, seed)
        bg = torch.full_like(target, 0.5)
        res.canvases = torch.stack(
            [bg + (target - bg) * min(1.0, n / self.speed) for n in res.checkpoints]
        )
        return res


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("run")
    images = [context(f"img{i:02d}", i) for i in range(10)]
    producers = [
        Converging("fake", 2500, 1800),
        Converging("slow", 2500, 4000, dagger=("l1",)),
        Converging("short", 600, 500),  # cannot reach 1000 or 2000 strokes
    ]
    rn.run_harness(
        producers,
        images,
        {"fake": [0, 1], "slow": [0], "short": [0]},
        PROTO,
        out,
        timed=False,
        log=lambda _: None,
    )
    return out


def results(run_dir: Path, proto: Any = PROTO) -> rp.Results:
    return rp.Results(rp.load_records(run_dir), rp.load_controls(run_dir), proto)


def test_tables_have_one_row_per_method_and_mark_unreachable_budgets(run_dir: Path) -> None:
    res = results(run_dir)
    assert res.methods == ["fake", "short", "slow"]
    b1 = rp.table_b1(res)
    assert "n/a (max n = 600)" in b1  # the short producer cannot reach 1000 and 2000 strokes
    assert (
        "L1 †" in b1 and b1.count("L1 †") == 1
    )  # the dagger only for the method whose loss has L1 (slow)
    b2, b3, b5 = rp.table_b2(res), rp.table_b3(res), rp.table_b5(res)
    assert "AUC@2000" in b2 and "n/a (max n = 600)" in b2 and "SW-AUC@2000" in b3 and "short" in b5
    assert "blank_black" in rp.table_controls(res) and "native blank canvas" in rp.table_controls(
        res
    )


def test_replicates_are_averaged_per_image_and_the_noise_is_reported_apart(run_dir: Path) -> None:
    res = results(run_dir)
    per = res.per_image("fake", res.metric("lpips_alex", 250))
    assert set(per) == {f"img{i:02d}" for i in range(10)}
    assert all(
        std == pytest.approx(0.0, abs=1e-9) for _, std in per.values()
    )  # deterministic producer: no replicate noise
    assert "±" in rp.describe_cell(per, PROTO)


def test_the_f1_family_is_holm_corrected_and_the_faster_method_wins(run_dir: Path) -> None:
    res = results(run_dir)
    text = rp.stats_tables(res)
    assert "F1 (confirmatory" in text and "fake vs slow" in text and "Exploratory" in text
    pairs = {"fake vs short", "fake vs slow", "short vs slow"}
    assert all(p in text for p in pairs)
    line = next(ln for ln in text.splitlines() if ln.startswith("| fake vs slow | P1"))
    assert (
        "| yes |" in line
    )  # fake converges faster than slow on every image: p Holm small, |r| = 1, CI excludes 0
    short_line = next(ln for ln in text.splitlines() if ln.startswith("| fake vs short | P2"))
    assert "n/a" in short_line  # the short producer has no AUC@2000: excluded, never imputed


def test_strokes_to_target_table_and_the_tau_rule(run_dir: Path) -> None:
    res = results(run_dir)
    tau = eval_run.compute_tau(res, PROTO)
    assert tau["strongest_baseline"] == "fake"  # the lowest dev mean LPIPS-Alex at 1000 strokes
    assert set(tau["values"]) == {500, 1000, 2000}  # type: ignore[arg-type]
    assert tau["values"][500] > tau["values"][1000] > tau["values"][2000]  # type: ignore[index]
    with_tau = dataclasses.replace(
        PROTO, tau={"status": "x", "strongest_baseline": "fake", "values": tau["values"]}
    )
    b4 = rp.table_b4(rp.Results(res.records, res.controls, with_tau))
    assert "tau_1000" in b4 and "100 %" in b4 or "%" in b4
    unset = dataclasses.replace(
        PROTO,
        tau={
            "status": "x",
            "strongest_baseline": None,
            "values": {500: None, 1000: None, 2000: None},
        },
    )
    assert "is not set" in rp.table_b4(rp.Results(res.records, res.controls, unset))


def test_tau_refuses_non_dev_records(run_dir: Path) -> None:
    res = results(run_dir)
    res.records[0] = {**res.records[0], "split": "test"}
    with pytest.raises(SystemExit, match="dev records only"):
        eval_run.compute_tau(res, PROTO)


def test_the_figure_is_written_small_and_with_tau(run_dir: Path, tmp_path: Path) -> None:
    res = results(run_dir)
    taus = eval_run.compute_tau(res, PROTO)["values"]
    proto = dataclasses.replace(
        PROTO, tau={"status": "x", "strongest_baseline": "fake", "values": taus}
    )
    path = rp.figure(rp.Results(res.records, res.controls, proto), tmp_path / "fig.png")
    assert path.exists() and 5_000 < path.stat().st_size < 500_000
    assert SIZE == 64


def test_the_runtime_table_leaves_out_the_warm_up_image(run_dir: Path) -> None:
    res = results(run_dir)
    b5 = rp.table_b5(res)
    row = next(ln for ln in b5.splitlines() if ln.startswith("| fake |"))
    assert "18 of 20" in row  # 10 images x 2 seeds, the first image of each seed is the warm-up
    short = next(ln for ln in b5.splitlines() if ln.startswith("| short |"))
    assert "9 of 10" in short
    old = res.records[0]
    assert res.is_warmup({k: v for k, v in old.items() if k != "warmup"}) == (
        old["image_id"] == min(res.ok[old["method"]])
    )  # records written before the flag existed: the first image in sorted order
