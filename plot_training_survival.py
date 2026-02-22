from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_RANK_REWARD_TABLE = {
    1: 1.00,
    2: 0.60,
    3: 0.25,
    4: 0.00,
    5: -0.25,
    6: -0.60,
}


def load_rows(metrics_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    acc = 0.0
    q: List[float] = []
    for v in values:
        q.append(float(v))
        acc += float(v)
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def rank_to_reward(rank: float, table: Dict[int, float]) -> float:
    if rank <= 1.0:
        return table[1]
    if rank >= 6.0:
        return table[6]
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return table[lo]
    w = rank - lo
    return table[lo] * (1.0 - w) + table[hi] * w


def load_rank_reward_table(metrics_path: Path) -> Dict[int, float]:
    train_cfg = metrics_path.parent / "train_config.json"
    if not train_cfg.exists():
        return dict(DEFAULT_RANK_REWARD_TABLE)
    try:
        data = json.loads(train_cfg.read_text(encoding="utf-8"))
        rewards = data.get("rewards", {})
        return {
            1: float(rewards.get("first", DEFAULT_RANK_REWARD_TABLE[1])),
            2: float(rewards.get("second", DEFAULT_RANK_REWARD_TABLE[2])),
            3: float(rewards.get("third", DEFAULT_RANK_REWARD_TABLE[3])),
            4: float(rewards.get("fourth", DEFAULT_RANK_REWARD_TABLE[4])),
            5: float(rewards.get("fifth", DEFAULT_RANK_REWARD_TABLE[5])),
            6: float(rewards.get("sixth", DEFAULT_RANK_REWARD_TABLE[6])),
        }
    except Exception:
        return dict(DEFAULT_RANK_REWARD_TABLE)


def extract_survival_series(rows: Sequence[Dict[str, object]]) -> Tuple[List[int], List[float]]:
    updates = [int(row["update"]) for row in rows]
    values = [float(row["avg_survival_steps"]) for row in rows]
    return updates, values


def extract_placement_series(
    rows: Sequence[Dict[str, object]],
    rank_reward_table: Dict[int, float],
) -> Tuple[List[int], List[float]]:
    # Placement score is a normalized rank-objective score in [0,100],
    # where 1st place maps to 100 and 6th maps to 0.
    updates = [int(row["update"]) for row in rows]
    rewards = [rank_to_reward(float(row["avg_rank"]), rank_reward_table) for row in rows]
    lo = rank_reward_table[6]
    hi = rank_reward_table[1]
    denom = max(1e-9, hi - lo)
    score = [((r - lo) / denom) * 100.0 for r in rewards]
    return updates, score


def render_svg(
    updates: Sequence[int],
    values: Sequence[float],
    ma: Sequence[float],
    output_path: Path,
    title: str,
    y_label: str,
    raw_label: str,
    ma_label: str,
    ma_window: int,
) -> None:
    width = 1200
    height = 700
    margin_l = 95
    margin_r = 40
    margin_t = 70
    margin_b = 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    xmin = min(updates)
    xmax = max(updates)
    ymin = min(min(values), min(ma))
    ymax = max(max(values), max(ma))
    pad = max(1e-6, (ymax - ymin) * 0.08)
    ymin -= pad
    ymax += pad
    if ymax <= ymin:
        ymax = ymin + 1.0

    def sx(x: float) -> float:
        if xmax == xmin:
            return margin_l + plot_w / 2.0
        return margin_l + ((x - xmin) / (xmax - xmin)) * plot_w

    def sy(y: float) -> float:
        return margin_t + plot_h - ((y - ymin) / (ymax - ymin)) * plot_h

    def polyline(xs: Sequence[int], ys: Sequence[float]) -> str:
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))

    xticks = 10
    yticks = 8
    grid = []
    labels = []

    for i in range(xticks + 1):
        xval = xmin + (xmax - xmin) * (i / xticks)
        xpix = sx(xval)
        grid.append(
            f'<line x1="{xpix:.2f}" y1="{margin_t}" x2="{xpix:.2f}" y2="{margin_t + plot_h}" '
            f'stroke="#e8e8e8" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{xpix:.2f}" y="{height - 34}" text-anchor="middle" font-size="14" fill="#444">'
            f"{int(round(xval))}</text>"
        )

    for i in range(yticks + 1):
        yval = ymin + (ymax - ymin) * (i / yticks)
        ypix = sy(yval)
        grid.append(
            f'<line x1="{margin_l}" y1="{ypix:.2f}" x2="{margin_l + plot_w}" y2="{ypix:.2f}" '
            f'stroke="#e8e8e8" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{margin_l - 12}" y="{ypix + 5:.2f}" text-anchor="end" font-size="14" fill="#444">{yval:.2f}</text>'
        )

    raw_points = polyline(updates, values)
    ma_points = polyline(updates, ma)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{width/2:.1f}" y="36" text-anchor="middle" font-size="24" fill="#111" font-family="Arial,sans-serif">{title}</text>
  {"".join(grid)}
  <line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#222" stroke-width="2"/>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#222" stroke-width="2"/>
  {"".join(labels)}
  <polyline fill="none" stroke="#7ea6e0" stroke-width="2" opacity="0.75" points="{raw_points}"/>
  <polyline fill="none" stroke="#0f4fa8" stroke-width="3" points="{ma_points}"/>
  <rect x="{width - 420}" y="78" width="390" height="74" fill="#fff" stroke="#bbb"/>
  <line x1="{width - 400}" y1="102" x2="{width - 330}" y2="102" stroke="#7ea6e0" stroke-width="3"/>
  <text x="{width - 320}" y="107" font-size="14" fill="#222">{raw_label}</text>
  <line x1="{width - 400}" y1="128" x2="{width - 330}" y2="128" stroke="#0f4fa8" stroke-width="4"/>
  <text x="{width - 320}" y="133" font-size="14" fill="#222">{ma_label} (MA {ma_window})</text>
  <text x="{width/2:.1f}" y="{height - 10}" text-anchor="middle" font-size="16" fill="#222">Training update</text>
  <text transform="translate(28,{height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="16" fill="#222">{y_label}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def write_metric_plot(
    metrics_path: Path,
    output_path: Path,
    *,
    metric: str,
    ma_window: int = 10,
) -> Path:
    rows = load_rows(metrics_path)
    if not rows:
        raise RuntimeError(f"No metric rows found in {metrics_path}")

    if metric == "survival":
        updates, values = extract_survival_series(rows)
        title = "Survival Progress vs Training Update"
        y_label = "Average survival steps"
        raw_label = "Avg survival (raw)"
        ma_label = "Avg survival"
    elif metric == "placement":
        table = load_rank_reward_table(metrics_path)
        updates, values = extract_placement_series(rows, table)
        title = "Placement Progress vs Training Update"
        y_label = "Placement score (0-100, rank-first)"
        raw_label = "Placement score (raw)"
        ma_label = "Placement score"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    ma = moving_average(values, ma_window)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".svg":
        render_svg(updates, values, ma, output_path, title, y_label, raw_label, ma_label, ma_window)
        return output_path

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        fallback = output_path.with_suffix(".svg")
        render_svg(updates, values, ma, fallback, title, y_label, raw_label, ma_label, ma_window)
        return fallback

    plt.figure(figsize=(11, 6))
    plt.plot(updates, values, color="#7ea6e0", linewidth=1.5, alpha=0.65, label=f"{raw_label}")
    plt.plot(updates, ma, color="#0f4fa8", linewidth=2.5, label=f"{ma_label} (MA {ma_window})")
    plt.xlabel("Training update")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    return output_path


def write_default_plots(metrics_path: Path, output_dir: Path, ma_window: int = 10) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    survival_path = write_metric_plot(
        metrics_path,
        output_dir / "survival_progress.svg",
        metric="survival",
        ma_window=ma_window,
    )
    placement_path = write_metric_plot(
        metrics_path,
        output_dir / "placement_progress.svg",
        metric="placement",
        ma_window=ma_window,
    )
    return {"survival": survival_path, "placement": placement_path}


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training metrics vs update.")
    p.add_argument("--metrics", required=True, type=str, help="Path to metrics.jsonl")
    p.add_argument(
        "--output",
        required=False,
        type=str,
        default="",
        help="Output image path (.svg or .png). Defaults to metric-specific SVG in metrics dir.",
    )
    p.add_argument("--metric", choices=["survival", "placement"], default="survival")
    p.add_argument("--ma-window", type=int, default=10, help="Moving-average window in updates")
    p.add_argument(
        "--write-default-set",
        action="store_true",
        help="Write both survival_progress.svg and placement_progress.svg into metrics directory",
    )
    args = p.parse_args()

    metrics_path = Path(args.metrics)
    if args.write_default_set:
        out = write_default_plots(metrics_path, metrics_path.parent, ma_window=args.ma_window)
        print(f"Wrote {out['survival']}")
        print(f"Wrote {out['placement']}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        filename = "survival_progress.svg" if args.metric == "survival" else "placement_progress.svg"
        output_path = metrics_path.parent / filename

    written = write_metric_plot(metrics_path, output_path, metric=args.metric, ma_window=args.ma_window)
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
