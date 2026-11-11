"""Run the member-two pipeline in order.

    python scripts/run_all.py                 # v2 (RMSE) pipeline
    python scripts/run_all.py --fast         # skip the heavy model/scaling steps
    python scripts/run_all.py --only itemcf-rating report-v2
    python scripts/run_all.py --legacy       # v1 NDCG / Top-N demo chain

Two protocols live in this repo:

* **v2 (default, official task)** — predict a rating for the 100,000
  ``test_pairs`` and score RMSE. This is the main line.
* **v1 (``--legacy``)** — the old NDCG@10 / Top-10 candidate-ranking chain,
  kept as a demo experiment. Its numbers are obsolete for grading.

Every step shells out to the same script README documents; there is no second
implementation hidden in here. A failing step stops the run unless
``--keep-going`` is given.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def v2_steps() -> list[tuple[str, str, list[str], bool]]:
    """(key, label, argv, heavy)"""
    return [
        ("tests", "单元测试（36 项）", [PY, "-m", "pytest", "tests", "-q"], False),
        ("verify", "Step 1 数据验收 + 画像",
         [PY, "scripts/verify_hybrid_data.py"], False),
        ("val-split", "Step 2 本地 val 切分",
         [PY, "scripts/make_val_split.py"], True),
        ("eval-check", "Step 3 RMSE 评测器自检",
         [PY, "scripts/check_evaluator.py"], False),
        ("baselines", "Step 4 基线阶梯 + 正式 B0",
         [PY, "scripts/run_baselines.py"], True),
        ("itemcf-rating", "Step 5a ItemCF 评分预测",
         [PY, "scripts/run_itemcf_rating.py"], True),
        ("lsh-rating", "Step 5b LSH 近邻加速（9 组参数）",
         [PY, "scripts/run_lsh_rating.py"], True),
        ("als", "Step 6a Spark ALS 调参 + 全量",
         [PY, "scripts/run_als.py"], True),
        ("als-scaling", "Step 6b ALS 扩展性 1/2/4/8",
         [PY, "scripts/run_als.py", "--scaling"], True),
        ("predict-test", "Step 7 test 预测 + pred_val",
         [PY, "scripts/predict_test.py", "--model", "auto"], True),
        ("report-v2", "v2 交付表格",
         [PY, "scripts/report_v2.py"], False),
        ("plot", "ALS 扩展性曲线",
         [PY, "scripts/plot_scaling.py", "--als"], False),
    ]


def v1_steps() -> list[tuple[str, str, list[str], bool]]:
    """Retired NDCG chain, kept runnable for the report's demo section."""
    return [
        ("v1-popular", "v1 热门基线 B0",
         [PY, "scripts/run_popular.py"], True),
        ("v1-itemcf", "v1 精确 ItemCF（NDCG）",
         [PY, "scripts/run_itemcf.py"], True),
        ("v1-lsh", "v1 LSH 参数扫描（NDCG）",
         [PY, "scripts/run_lsh.py"], True),
        ("v1-spark", "v1 Spark ItemCF",
         [PY, "scripts/run_spark_itemcf.py", "--stage", "val", "--topk", "200"], True),
        ("v1-tables", "v1 表格",
         [PY, "scripts/report_tables.py"], False),
        ("v1-plot", "v1 曲线",
         [PY, "scripts/plot_scaling.py"], False),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip heavy steps")
    parser.add_argument("--legacy", action="store_true", help="run the v1 NDCG demo chain")
    parser.add_argument("--only", nargs="*", default=None, help="step keys to run")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    all_steps = v1_steps() + v2_steps()
    plan = v1_steps() if args.legacy else v2_steps()
    if args.only:
        keys = set(args.only)
        plan = [s for s in plan if s[0] in keys]
        unknown = keys - {s[0] for s in all_steps}
        if unknown:
            print(f"unknown step(s): {', '.join(sorted(unknown))}")
            print("available:", ", ".join(s[0] for s in all_steps))
            return 2
    elif args.fast:
        plan = [s for s in plan if not s[3]]

    started = time.perf_counter()
    failures: list[str] = []
    for index, (key, label, argv, _heavy) in enumerate(plan, start=1):
        print(f"\n{'=' * 68}\n[{index}/{len(plan)}] {label}  ({key})\n{'=' * 68}", flush=True)
        t0 = time.perf_counter()
        done = subprocess.run(argv, cwd=ROOT)
        elapsed = time.perf_counter() - t0
        if done.returncode != 0:
            failures.append(key)
            print(f"FAILED {key} after {elapsed:.1f}s", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"OK {key} in {elapsed:.1f}s", flush=True)

    print(f"\ntotal {time.perf_counter() - started:.1f}s")
    if failures:
        print("failed steps:", ", ".join(failures))
        return 1
    print("ALL STEPS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
