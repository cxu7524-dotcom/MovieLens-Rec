"""Collect the result files into the three tables member four asked for.

    python scripts/report_tables.py

Reads only ``results/*.json`` (nothing is recomputed) and writes

  results/table_quality.md      NDCG@10 / Recall@10 / overlap@100 per model
  results/table_efficiency.md   wall clock, peak memory, per-user latency
  results/table_scaling.md      Spark parallelism 1/2/4/8

plus the same content as CSV for pasting into the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join("" if c is None else str(c) for c in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value, digits: int = 4, default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    settings = load_settings()
    out = settings.project_root / "results"

    random_ = load(out / "metrics_random_val.json")
    popular = load(out / "metrics_popular_val.json")
    popular_all = load(out / "baseline_popular.json")
    itemcf = load(out / "itemcf_val.json")
    itemcf_metrics = load(out / "metrics_itemcf_val.json")
    lsh = load(out / "lsh_val.json")
    spark_val = load(out / "spark_itemcf_val_p16.json") or load(out / "spark_itemcf_val_p8.json")
    scaling = load(out / "spark_scaling_val.json")

    # ---------------- quality ----------------
    headers = ["模型", "配置", "NDCG@10", "Recall@10", "覆盖率", "overlap@100"]
    rows: list[list] = []
    if random_:
        rows.append(["random", "均匀随机", fmt(random_["ndcg@10"]), fmt(random_["recall@10"]),
                     fmt(random_["coverage"]), "-"])
    if popular:
        rows.append(["popular", "全局正反馈计数 (B0)", fmt(popular["ndcg@10"]),
                     fmt(popular["recall@10"]), fmt(popular["coverage"]), "-"])
    if popular_all:
        for variant in popular_all.get("variants", []):
            if variant["params"].get("kind") != "decay":
                continue
            rows.append([
                "popular-decay",
                f"半衰期 {int(variant['params']['half_life_days'])} 天",
                fmt(variant["ndcg@10"]), fmt(variant["recall@10"]),
                fmt(variant["coverage"]), "-",
            ])
    if itemcf:
        for entry in itemcf.get("k_sweep", []):
            rows.append(["itemcf (精确)", f"K={entry['k_neighbors']}，20k 用户抽样",
                         fmt(entry["ndcg@10"]), fmt(entry["recall@10"]),
                         fmt(entry["coverage"]), "1.0000"])
        full = itemcf["full"]
        rows.append(["itemcf (精确)", f"K={itemcf['k_chosen']}，全量",
                     fmt(full["ndcg@10"]), fmt(full["recall@10"]),
                     fmt(full["coverage"]), "1.0000"])
    if lsh:
        for entry in lsh.get("sweep", []):
            tag = "LSH" if entry["group"] == "grid" else "LSH(诊断)"
            rows.append([
                tag,
                f"bands={entry['bands']}, rows={entry['rows']}, 20k 用户抽样",
                fmt(entry["sample_ndcg@10"]), fmt(entry["sample_recall@10"]),
                fmt(entry["sample_coverage"]), fmt(entry["overlap@100"]),
            ])
        delivered = lsh["delivered"]
        chosen = next((e for e in lsh["sweep"]
                       if e["bands"] == lsh["config"]["bands"]
                       and e["rows"] == lsh["config"]["rows"]), {})
        rows.append([
            "LSH (交付)", f"bands={lsh['config']['bands']}, rows={lsh['config']['rows']}，全量",
            fmt(delivered["ndcg@10"]), fmt(delivered["recall@10"]),
            fmt(delivered["coverage"]), fmt(chosen.get("overlap@100")),
        ])
    if spark_val:
        m = spark_val["metrics"]
        rows.append(["itemcf (Spark)", f"topk={spark_val['topk']}, local[{spark_val['parallelism']}]",
                     fmt(m.get("ndcg@10")), fmt(m.get("recall@10")), fmt(m.get("coverage")), "-"])

    quality = "# 近似质量表（val 阶段）\n\n" + md_table(headers, rows) + "\n"
    (out / "table_quality.md").write_text(quality, encoding="utf-8")
    write_csv(out / "table_quality.csv", headers, rows)

    # ---------------- efficiency ----------------
    headers = ["模型", "相似度/生成 s", "打分 s", "峰值内存 MB", "单用户延迟 ms", "吞吐 pairs/s"]
    rows = []
    if popular:
        count_variant = next(
            (v for v in (popular_all or {}).get("variants", [])
             if v.get("name") == "popular_count_val"),
            {},
        )
        score_s = count_variant.get("score_s") or popular["elapsed_s"]
        rows.append(["popular", f"{score_s:.1f} (查表)", fmt(score_s, 1),
                     "-", fmt(1000 * score_s / max(popular["users"], 1), 3),
                     fmt(popular["pairs"] / max(score_s, 1e-9), 0)])
    if itemcf:
        full = itemcf["full"]
        rows.append(["itemcf (精确)", fmt(itemcf["similarity"]["build_s"], 1),
                     fmt(full["score_s"], 1), fmt(full["peak_rss_mb"], 0),
                     fmt(1000 * full["score_s"] / max(full["users"], 1), 3),
                     fmt(full["pairs_per_s"], 0)])
    if lsh:
        d = lsh["delivered"]
        gen = next((e["gen_s"] for e in lsh["sweep"]
                    if e["bands"] == lsh["config"]["bands"] and e["rows"] == lsh["config"]["rows"]), None)
        rows.append(["LSH (交付)", fmt(gen, 1), fmt(d["score_s"], 1), fmt(d["peak_rss_mb"], 0),
                     fmt(1000 * d["score_s"] / max(d["users"], 1), 3),
                     fmt(d["pairs"] / max(d["score_s"], 1e-9), 0)])
    if spark_val:
        t = spark_val["timings"]
        scoring = (t.get("to_spark_s") or 0) + t["scoring_s"]
        rows.append([f"itemcf (Spark local[{spark_val['parallelism']}])",
                     fmt(t["cooccurrence_s"], 1), fmt(scoring, 1),
                     fmt(spark_val["peak_rss_mb"], 0),
                     fmt(1000 * scoring / max(spark_val["metrics"]["users"], 1), 3),
                     fmt(spark_val["pairs"] / max(scoring, 1e-9), 0)])

    efficiency = "# 效率对比表（val 阶段，20,281,548 个候选对）\n\n" + md_table(headers, rows) + "\n"
    if lsh:
        efficiency += (
            f"\n> LSH 另有固定开销：{lsh['signature_hashes']} 个哈希的 MinHash 签名构建 "
            f"{lsh['signature_s']} s（不随 bands/rows 变化）。\n"
        )
    (out / "table_efficiency.md").write_text(efficiency, encoding="utf-8")
    write_csv(out / "table_efficiency.csv", headers, rows)

    # ---------------- scaling ----------------
    if scaling:
        headers = ["并行度", "总耗时 s", "共现 s", "打分 s", "转换 s", "吞吐 pairs/s", "加速比", "并行效率"]
        rows = [[r["parallelism"], fmt(r["total_s"], 1), fmt(r["cooccurrence_s"], 1),
                 fmt(r["scoring_s"], 1), fmt(r.get("to_spark_s"), 1),
                 fmt(r["pairs_per_s"], 0), fmt(r["speedup"], 2),
                 fmt(r["efficiency"], 2)] for r in scaling["runs"]]
        table = (
            f"# Spark 扩展性（{scaling['stage']} 阶段，topk={scaling['topk']}，"
            f"block={scaling['block_items']}，固定 {scaling.get('users_sampled')} 用户抽样）\n\n"
            + md_table(headers, rows)
            + "\n\n> 加速比相对同系列 local[1] 运行；并行效率 = 加速比 / 并行度。"
              "「转换 s」是 pandas → Spark DataFrame 的固定开销，不随并行度变化。\n"
        )
        (out / "table_scaling.md").write_text(table, encoding="utf-8")
        write_csv(out / "table_scaling.csv", headers, rows)

    print(quality)
    print(efficiency)
    if scaling:
        print((out / "table_scaling.md").read_text(encoding="utf-8"))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
