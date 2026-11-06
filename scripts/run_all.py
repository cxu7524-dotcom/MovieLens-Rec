"""Run the whole member-two pipeline in order.

    python scripts/run_all.py               # everything (~70 min)
    python scripts/run_all.py --fast        # skip the LSH sweep and Spark (~5 min)
    python scripts/run_all.py --only itemcf # run one stage

Each step shells out to the same script the README documents, so there is one
code path whether a step is run by hand or from here. A failing step stops the
run unless ``--keep-going`` is given.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def steps() -> list[tuple[str, str, list[str], bool]]:
    """(key, label, argv, heavy)"""
    return [
        ("tests", "单元测试", [PY, "-m", "pytest", "tests", "-q"], False),
        ("eval", "评测器自检（随机分数）", [PY, "scripts/check_evaluator.py"], False),
        ("popular", "热门基线 B0", [PY, "scripts/run_popular.py"], False),
        ("itemcf", "精确 ItemCF（val）", [PY, "scripts/run_itemcf.py"], True),
        ("itemcf-test", "精确 ItemCF（test 分数文件）",
         [PY, "scripts/run_itemcf.py", "--stage", "test", "--k", "200"], True),
        ("lsh", "LSH 参数扫描（val）", [PY, "scripts/run_lsh.py"], True),
        ("lsh-test", "LSH（test 分数文件）", [PY, "scripts/run_lsh.py", "--stage", "test"], True),
        ("spark", "Spark ItemCF（val）",
         [PY, "scripts/run_spark_itemcf.py", "--stage", "val", "--topk", "200"], True),
        ("spark-test", "Spark ItemCF（test 分数文件）",
         [PY, "scripts/run_spark_itemcf.py", "--stage", "test", "--topk", "200",
          "--write-scores"], True),
        ("scaling", "Spark 扩展性 1/2/4/8/16",
         [PY, "scripts/run_spark_itemcf.py", "--scaling", "--stage", "val",
          "--topk", "200", "--users", "30000"], True),
        ("tables", "交付表格", [PY, "scripts/report_tables.py"], False),
        ("curve", "扩展性曲线", [PY, "scripts/plot_scaling.py"], False),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip heavy steps")
    parser.add_argument("--only", nargs="*", default=None, help="step keys to run")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    plan = steps()
    if args.only:
        plan = [s for s in plan if s[0] in set(args.only)]
        missing = set(args.only) - {s[0] for s in plan}
        if missing:
            print(f"unknown step(s): {', '.join(sorted(missing))}")
            print("available:", ", ".join(s[0] for s in steps()))
            return 2
    elif args.fast:
        plan = [s for s in plan if not s[3]]

    total_started = time.perf_counter()
    failures: list[str] = []
    for index, (key, label, argv, _heavy) in enumerate(plan, start=1):
        print(f"\n{'=' * 68}\n[{index}/{len(plan)}] {label}  ({key})\n{'=' * 68}", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(argv, cwd=ROOT)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            failures.append(key)
            print(f"FAILED {key} after {elapsed:.1f}s", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"OK {key} in {elapsed:.1f}s", flush=True)

    print(f"\ntotal {time.perf_counter() - total_started:.1f}s")
    if failures:
        print("failed steps:", ", ".join(failures))
        return 1
    print("ALL STEPS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
