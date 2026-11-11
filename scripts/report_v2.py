"""Step 7 support: assemble every v2 table member four needs.

    python scripts/report_v2.py

Reads only ``results/*.json`` — nothing is recomputed. Writes markdown + CSV
versions of:

  results/v2_table_models.md       all models on val, vs B0, per-half RMSE
  results/v2_table_efficiency.md   build/predict time, peak memory, throughput
  results/v2_table_params.md       ItemCF K/beta sweep, ALS 12-config grid, LSH grid
  results/v2_table_scaling.md      ALS 1/2/4/8 training time and speed-up

v1 (NDCG) tables are deliberately untouched: they live under their original
names and are kept as the Top-N demo experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402


def load(out_dir: Path, name: str) -> dict:
    path = out_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def write(out_dir: Path, stem: str, headers: list[str], rows: list[list],
          title: str, foot: str = "") -> None:
    (out_dir / f"{stem}.md").write_text(
        f"# {title}\n\n" + table(headers, rows) + ("\n" + foot + "\n" if foot else ""),
        encoding="utf-8")
    lines = [",".join(headers)] + [",".join("" if c is None else str(c) for c in r)
                                   for r in rows]
    (out_dir / f"{stem}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(value, digits: int = 6) -> str:
    return "-" if value is None else (f"{value:.{digits}f}" if isinstance(value, float)
                                     else str(value))


def main() -> int:
    settings = load_settings()
    out = settings.project_root / "results"
    base = load(out, "baselines_val.json")
    itemcf = load(out, "itemcf_rating_val.json")
    lsh = load(out, "lsh_rating_val.json")
    als = load(out, "als_val.json")
    scaling = load(out, "als_scaling_val.json")
    summary = load(out, "delivery_summary.json")
    b0 = base.get("b0_global_mean_rmse")

    # ---- model comparison ----
    rows = []
    for level in base.get("ladder", []):
        half = level.get("by_half", {})
        rows.append([level["level"], f(level["rmse"]), f(level["mae"]),
                     f((b0 - level["rmse"]) / b0 * 100, 2) if b0 else "-",
                     f(half.get("movielens32m", {}).get("rmse")),
                     f(half.get("synthetic", {}).get("rmse"))])
    full_itemcf = itemcf.get("full", {})
    if full_itemcf:
        rows.append(["itemcf-rating (K=%s, beta=%g)" % (itemcf.get("k_chosen"),
                                                        itemcf.get("beta_chosen")),
                     f(full_itemcf.get("rmse")), f(full_itemcf.get("mae")),
                     f((b0 - full_itemcf["rmse"]) / b0 * 100, 2) if b0 else "-",
                     f(full_itemcf.get("by_half", {}).get("movielens32m", {}).get("rmse")),
                     f(full_itemcf.get("by_half", {}).get("synthetic", {}).get("rmse"))])
    full_als = als.get("full", {})
    if full_als:
        rows.append(["spark-als (%s)" % als.get("best_config", {}).get("rank"),
                     f(full_als.get("rmse")), f(full_als.get("mae")),
                     f((b0 - full_als["rmse"]) / b0 * 100, 2) if b0 else "-",
                     f(full_als.get("by_half", {}).get("movielens32m", {}).get("rmse")),
                     f(full_als.get("by_half", {}).get("synthetic", {}).get("rmse"))])
    delivered_lsh = lsh.get("delivered", {})
    if delivered_lsh:
        rows.append(["lsh-neighbours (近似)", f(delivered_lsh.get("rmse")),
                     f(delivered_lsh.get("mae")),
                     f((b0 - delivered_lsh["rmse"]) / b0 * 100, 2) if b0 else "-",
                     f(delivered_lsh.get("by_half", {}).get("movielens32m", {}).get("rmse")),
                     f(delivered_lsh.get("by_half", {}).get("synthetic", {}).get("rmse"))])
    write(out, "v2_table_models",
          ["模型", "RMSE", "MAE", "相对 B0 提升 %", "ML 半区 RMSE", "合成半区 RMSE"],
          rows, "v2 评分预测模型对比（val，99,998 对，pair 等权）",
          f"B0 = 全局均值 = {f(b0)}；10% 相对提升目标 RMSE <= "
          f"{f(b0 * (1 - 0.10) if b0 else None)}（口径待老师确认）。")

    # ---- efficiency ----
    rows = []
    if itemcf.get("similarity"):
        s = itemcf["similarity"]
        rows.append(["itemcf-rating", f(s.get("build_s"), 1), "-",
                     f(itemcf.get("full", {}).get("elapsed_s"), 1),
                     f(s.get("peak_mb"), 0), f(itemcf.get("elapsed_s"), 1),
                     "相似度 142,608 部电影分块 1000"])
    if lsh.get("signature"):
        rows.append(["lsh-neighbours", f(lsh["signature"].get("build_s"), 1),
                     f(lsh.get("best_overlap_config", {}).get("gen_s"), 1), "-",
                     f(lsh.get("peak_rss_mb"), 0), f(lsh.get("elapsed_s"), 1),
                     "9 组参数；生成耗时取最优 overlap 配置"])
    if lsh.get("exact"):
        rows.append(["精确邻居（对照）", f(lsh["exact"].get("build_s"), 1), "-",
                     "-", "-", "-", f"{lsh['exact'].get('edges', 0):,} 条边"])
    if full_als:
        rows.append(["spark-als 全量", f(full_als.get("train_s"), 1), "-",
                     "-", "-", f(als.get("total_s"), 1),
                     f"{full_als.get('ratings', 0):,} 条评分，"
                     f"{full_als.get('throughput_ratings_per_s', 0):,.0f} 条/秒"])
    write(out, "v2_table_efficiency",
          ["模型", "构建/签名 s", "邻居生成 s", "预测 s", "峰值 MB", "总耗时 s", "备注"],
          rows, "v2 效率对比")

    # ---- parameter sweeps ----
    lines = ["# v2 参数表", "", "## ItemCF：K × beta（val RMSE）", "",
             table(["K", "beta", "RMSE", "MAE", "预测 s", "邻居边数"],
                   [[e["k"], e["beta"], f(e["rmse"]), f(e["mae"]),
                     f(e.get("predict_s"), 1), f"{e.get('edges', 0):,}"]
                    for e in itemcf.get("k_sweep", [])]),
             "## ALS：rank × iterations × regParam（10% 样本调参）", "",
             table(["rank", "iterations", "regParam", "RMSE", "MAE", "fit s"],
                   [[e.get("rank"), e.get("iterations"), e.get("reg_param"),
                     f(e.get("rmse")), f(e.get("mae")), f(e.get("fit_s"), 1)]
                    for e in als.get("tuning", {}).get("grid", [])]),
             "## LSH：bands × rows", "",
             table(["bands", "rows", "候选对", "overlap@100", "生成 s", "峰值 MB", "RMSE"],
                   [[e["bands"], e["rows"], f"{e['pairs']:,}", f(e["overlap@100"], 4),
                     f(e["gen_s"], 1), f"{e['peak_mb']:,.0f}", f(e["rmse"])]
                    for e in lsh.get("sweep", [])])]
    (out / "v2_table_params.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- scaling ----
    runs = scaling.get("runs", [])
    if runs:
        rows = [[r["parallelism"], f(r["train_s"], 1), f(r["total_s"], 1),
                 f(r["throughput_ratings_per_s"], 0), f(r["speedup"], 2),
                 f(r.get("efficiency"), 2), f(r["rmse"])] for r in runs]
        write(out, "v2_table_scaling",
              ["并行度", "训练 s", "总耗时 s", "吞吐 条/秒", "加速比", "并行效率", "val RMSE"],
              rows,
              f"v2 ALS 扩展性（{scaling.get('fraction', 0):.0%} 样本 = "
              f"{scaling.get('ratings', 0):,} 条评分）",
              "加速比相对同系列 local[1] 的训练耗时；并行效率 = 加速比 / 并行度。")

    print(f"wrote v2 tables to {out}")
    if summary:
        print(f"delivered model: {summary.get('model')} | "
              f"val_rmse={summary.get('val_rmse')} | B0={summary.get('b0_global_mean_rmse')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
