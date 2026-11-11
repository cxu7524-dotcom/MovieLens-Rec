"""Draw the Spark scaling curve as a standalone SVG (no plotting dependency).

    python scripts/plot_scaling.py

Reads ``results/spark_scaling_<stage>.json`` and writes
``results/scaling_curve.svg`` — two panels: wall clock (total vs the parallel
co-occurrence phase) and speed-up against the ideal linear line. Matplotlib is
deliberately not a dependency: an SVG keeps the repo installable with what the
course already requires, and it pastes into Word/PowerPoint at any size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402

W, H = 820, 360
PAD_L, PAD_R, PAD_T, PAD_B = 62, 24, 68, 52
FONT = "'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', sans-serif"
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
BLUE = "#2563eb"
ORANGE = "#d97706"
GRAY = "#9aa5b1"


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return out_lo
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def _panel(x0: float, width: float, runs: list[dict], title: str,
           total_label: str = "总耗时", second_label: str = "共现阶段") -> list[str]:
    """One chart: x = parallelism, y depends on the caller's series."""
    top, bottom = PAD_T, H - PAD_B
    out = [f'<text x="{x0 + width / 2:.0f}" y="{top - 22}" text-anchor="middle" '
           f'font-size="13" font-weight="600" fill="{INK}">{title}</text>']

    xs = [r["parallelism"] for r in runs]
    totals = [r["total_s"] for r in runs]
    coocs = [r["cooccurrence_s"] for r in runs]
    y_hi = max(totals) * 1.12
    y_lo = 0.0

    # grid + y ticks
    for i in range(5):
        value = y_lo + (y_hi - y_lo) * i / 4
        y = _scale(value, y_lo, y_hi, bottom, top)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
                   f'fill="{MUTED}">{value:.0f}</text>')
    out.append(f'<text x="{x0 - 44}" y="{top - 4}" font-size="10" fill="{MUTED}">秒</text>')

    def px(i: int) -> float:
        return _scale(xs[i], xs[0], xs[-1], x0 + 14, x0 + width - 14)

    # x axis
    out.append(f'<line x1="{x0}" y1="{bottom}" x2="{x0 + width}" y2="{bottom}" '
               f'stroke="{MUTED}" stroke-width="1"/>')
    for i, n in enumerate(xs):
        out.append(f'<text x="{px(i):.1f}" y="{bottom + 16}" text-anchor="middle" '
                   f'font-size="10" fill="{MUTED}">{n}</text>')
    out.append(f'<text x="{x0 + width / 2:.0f}" y="{bottom + 34}" text-anchor="middle" '
               f'font-size="10" fill="{MUTED}">并行度（local[N]）</text>')

    series_spec = [(totals, BLUE, total_label)]
    if all(v is not None for v in coocs):
        series_spec.append((coocs, ORANGE, second_label))
    for series, colour, label in series_spec:
        points = " ".join(f"{px(i):.1f},{_scale(v, y_lo, y_hi, bottom, top):.1f}"
                          for i, v in enumerate(series))
        out.append(f'<polyline points="{points}" fill="none" stroke="{colour}" '
                   f'stroke-width="2" stroke-linejoin="round"/>')
        for i, v in enumerate(series):
            cx, cy = px(i), _scale(v, y_lo, y_hi, bottom, top)
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="#fff" '
                       f'stroke="{colour}" stroke-width="2"/>')
            out.append(f'<text x="{cx:.1f}" y="{cy - 9:.1f}" text-anchor="middle" '
                       f'font-size="9" fill="{colour}">{v:.0f}</text>')

    # legend, bottom-right where the falling curves leave room
    ly = bottom - 14
    for j, (colour, label) in enumerate(
            [(BLUE, total_label)] + ([(ORANGE, second_label)]
                                    if len(series_spec) > 1 else [])):
        lx = x0 + width - 178 + j * 92
        out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{colour}" '
                   f'stroke-width="2"/>')
        out.append(f'<text x="{lx + 23}" y="{ly + 4}" font-size="10" fill="{INK}">{label}</text>')
    return out


def _speedup_panel(x0: float, width: float, runs: list[dict]) -> list[str]:
    top, bottom = PAD_T, H - PAD_B
    xs = [r["parallelism"] for r in runs]
    speedups = [r["speedup"] for r in runs]
    # Zoomed to the actual range: plotting 0..cores would flatten a 1.0 -> 1.5
    # curve into the bottom 9% of the panel and hide the whole story.
    y_hi = max(speedups) * 1.35

    out = [f'<text x="{x0 + width / 2:.0f}" y="{top - 22}" text-anchor="middle" '
           f'font-size="13" font-weight="600" fill="{INK}">加速比（相对 local[1]）</text>']
    for i in range(5):
        value = y_hi * i / 4
        y = _scale(value, 0, y_hi, bottom, top)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
                   f'fill="{MUTED}">{value:.1f}×</text>')

    def px(n: float) -> float:
        return _scale(n, xs[0], xs[-1], x0 + 14, x0 + width - 14)

    out.append(f'<line x1="{x0}" y1="{bottom}" x2="{x0 + width}" y2="{bottom}" '
               f'stroke="{MUTED}" stroke-width="1"/>')
    for n in xs:
        out.append(f'<text x="{px(n):.1f}" y="{bottom + 16}" text-anchor="middle" '
                   f'font-size="10" fill="{MUTED}">{n}</text>')
    out.append(f'<text x="{x0 + width / 2:.0f}" y="{bottom + 34}" text-anchor="middle" '
               f'font-size="10" fill="{MUTED}">并行度</text>')

    y_one = _scale(1.0, 0, y_hi, bottom, top)
    out.append(f'<line x1="{x0}" y1="{y_one:.1f}" x2="{x0 + width}" y2="{y_one:.1f}" '
               f'stroke="{GRAY}" stroke-width="1.2" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{x0 + width - 6}" y="{y_one - 5:.1f}" text-anchor="end" '
               f'font-size="9" fill="{GRAY}">不加速（1.0×）</text>')
    out.append(f'<text x="{x0 + width - 6}" y="{top + 12}" text-anchor="end" font-size="9" '
               f'fill="{GRAY}">理想线性 16× 在图外</text>')

    points = " ".join(f"{px(n):.1f},{_scale(s, 0, y_hi, bottom, top):.1f}"
                      for n, s in zip(xs, speedups))
    out.append(f'<polyline points="{points}" fill="none" stroke="{BLUE}" stroke-width="2.2"/>')
    for n, s in zip(xs, speedups):
        cx, cy = px(n), _scale(s, 0, y_hi, bottom, top)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="#fff" stroke="{BLUE}" '
                   f'stroke-width="2"/>')
        out.append(f'<text x="{cx:.1f}" y="{cy - 9:.1f}" text-anchor="middle" font-size="10" '
                   f'fill="{BLUE}">{s:.2f}×</text>')
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--als", action="store_true",
                        help="plot the v2 ALS scaling series instead of the v1 ItemCF one")
    args = parser.parse_args()
    settings = load_settings()
    out_dir = settings.project_root / "results"
    path = out_dir / ("als_scaling_val.json" if args.als else "spark_scaling_val.json")
    if not path.is_file():
        print(f"missing {path}; run the scaling experiment first")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    if args.als:
        runs = [{
            "parallelism": r["parallelism"],
            "total_s": r["train_s"],
            "cooccurrence_s": None,
            "speedup": r["speedup"],
            "pairs_per_s": r["throughput_ratings_per_s"],
        } for r in data["runs"]]
        cfg = data.get("config", {})
        title = (f'Spark ALS 扩展性（{data["ratings"]:,} 条评分 = '
                 f'{data["fraction"]:.0%} 样本，rank={cfg.get("rank")} '
                 f'iter={cfg.get("iterations")} reg={cfg.get("reg_param")}）')
        total_label = "训练耗时"
        target = out_dir / "scaling_curve_als.svg"
    else:
        runs = data["runs"]
        title = (f'Spark ItemCF 扩展性（val，{data.get("users_sampled"):,} 用户抽样，'
                 f'topk={data["topk"]}，每块 {data["block_items"]} 部电影）')
        total_label = "总耗时"
        target = out_dir / "scaling_curve.svg"

    left_w = (W - PAD_L - PAD_R) * 0.54
    right_w = (W - PAD_L - PAD_R) - left_w - 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="24" font-size="14" font-weight="700" '
        f'fill="{INK}">{title}</text>',
    ]
    parts += _panel(PAD_L, left_w, runs,
                    "各阶段耗时" if not args.als else "训练耗时",
                    total_label=total_label)
    parts += _speedup_panel(PAD_L + left_w + 40, right_w, runs)
    parts.append("</svg>")

    target.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {target}")
    for row in runs:
        print(f"  local[{row['parallelism']:>2}] total {row['total_s']:>6.1f}s "
              f"cooc {row['cooccurrence_s']:>6.1f}s speedup {row['speedup']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
