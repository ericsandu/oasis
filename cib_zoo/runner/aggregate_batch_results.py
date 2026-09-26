"""Aggregator and Publication Dashboard Generator for CIB Batch Experiments.

Ingests all experiment JSON telemetry files from a batch run directory and produces:
1. cib_batch_summary.json: Structured tabular metrics and cross-preset comparisons.
2. cib_batch_dashboard.png: High-resolution 6-panel publication visualization.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cib_zoo.aggregator")


def load_batch_runs(batch_dir: Path) -> list[dict[str, Any]]:
    """Recursively discover and load all JSON telemetry files in the batch directory."""
    runs: list[dict[str, Any]] = []
    patterns = ["**/run_*.json", "**/cib_*.json", "**/*result*.json", "**/test_*.json"]
    candidates = []
    for pat in patterns:
        candidates.extend(batch_dir.glob(pat))
    # Deduplicate while preserving order
    seen = set()
    unique_files = [f for f in candidates if not (f in seen or seen.add(f))]
    for json_file in unique_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("status") == "success":
                    data["_file_path"] = str(json_file)
                    runs.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read {json_file}: {e}")
    return runs


def generate_summary_data(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compile structured metrics and comparisons across all runs."""
    summary_table = []
    by_preset: dict[str, list[dict[str, Any]]] = {}
    by_ratio: dict[float, list[dict[str, Any]]] = {}
    by_topology: dict[str, list[dict[str, Any]]] = {}

    for r in runs:
        preset = r.get("preset", "unknown")
        topology = r.get("topology", "unknown")
        topic_mode = r.get("topic_mode", "unknown")
        n_bots = r.get("num_bots", 0)
        bot_ratio = r.get("bot_ratio", 0.0)
        diff_amp = r.get("differential_amplification", 0.0)
        e_pay = r.get("exposure_payload", 0.0)
        e_base = r.get("exposure_baseline", 0.0)
        net_lift = e_pay - e_base
        bot_actions = r.get("telemetry", {}).get("total_bot_actions", 0)
        cost_eff = (net_lift / bot_actions) if bot_actions > 0 else 0.0

        entry = {
            "preset": preset,
            "campaign_name": r.get("campaign_name", ""),
            "topology": topology,
            "topic_mode": topic_mode,
            "num_organic": r.get("num_organic", 0),
            "num_bots": n_bots,
            "bot_ratio": bot_ratio,
            "exposure_baseline": e_base,
            "exposure_payload": e_pay,
            "net_lift": net_lift,
            "differential_amplification": diff_amp,
            "total_bot_actions": bot_actions,
            "cost_efficiency_per_action": cost_eff,
            "total_posts": r.get("telemetry", {}).get("total_posts", 0),
            "total_likes": r.get("telemetry", {}).get("total_likes", 0),
            "total_comments": r.get("telemetry", {}).get("total_comments", 0),
            "community_telemetry": r.get("community_telemetry", {}),
        }
        summary_table.append(entry)

        by_preset.setdefault(preset, []).append(entry)
        by_ratio.setdefault(round(bot_ratio, 2), []).append(entry)
        by_topology.setdefault(topology, []).append(entry)

    return {
        "total_runs": len(runs),
        "summary_table": summary_table,
        "by_preset": by_preset,
        "by_ratio": {str(k): v for k, v in by_ratio.items()},
        "by_topology": by_topology,
    }


def render_dashboard(runs: list[dict[str, Any]], output_png: Path) -> None:
    """Render 6-panel publication visualization."""
    if not HAS_MATPLOTLIB:
        logger.warning(
            "Matplotlib not installed in this environment; skipping PNG rendering."
        )
        return
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    fig.suptitle(
        "CIB Propagation & Algorithmic Amplification Batch Evaluation",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # 1. Panel A: Differential Amplification vs Bot Ratio
    ax1 = axes[0, 0]
    preset_colors = {
        "baseline": "#7f7f7f",
        "s1": "#1f77b4",
        "s2": "#ff7f0e",
        "s3": "#2ca02c",
    }
    preset_markers = {"s1": "o", "s2": "s", "s3": "^", "baseline": "x"}

    for p in ["s1", "s2", "s3"]:
        pts = [
            (r["bot_ratio"], r["differential_amplification"])
            for r in runs
            if r["preset"] == p and r.get("topic_mode") == "existing"
        ]
        if pts:
            pts.sort(key=lambda x: x[0])
            xs = [x[0] * 100 for x in pts]
            ys = [x[1] for x in pts]
            ax1.plot(
                xs,
                ys,
                marker=preset_markers.get(p, "o"),
                label=f"Preset {p.upper()}",
                color=preset_colors.get(p),
                linewidth=2.5,
                markersize=8,
            )

    ax1.set_title(
        r"A. Differential Amplification $\Delta\mathcal{A}$ vs. Bot Ratio",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Bot Infiltration Ratio (%)", fontsize=10)
    ax1.set_ylabel(r"Causal Lift per Bot (Net Exposure / $N_{bots}$)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Panel B: Cumulative Exposure Timeline
    ax2 = axes[0, 1]
    # Pick the representative S1 20% or longest run
    timeline_runs = [
        r
        for r in runs
        if r.get("exposure_timeline") and r.get("preset") in ("s1", "s3")
    ]
    if timeline_runs:
        best_run = max(timeline_runs, key=lambda x: len(x.get("exposure_timeline", [])))
        tl = best_run["exposure_timeline"]
        steps = [t["step"] for t in tl]
        pay_exp = [t["payload_exposure"] for t in tl]
        base_exp = [t["baseline_exposure"] for t in tl]
        ax2.plot(
            steps,
            pay_exp,
            marker="o",
            color="#d62728",
            label=f"Payload ({best_run['preset'].upper()} {int(best_run['bot_ratio'] * 100)}% bots)",
            linewidth=2.5,
        )
        ax2.plot(
            steps,
            base_exp,
            marker="s",
            color="#1f77b4",
            linestyle="--",
            label="Organic Baseline",
            linewidth=2,
        )
        ax2.fill_between(
            steps,
            base_exp,
            pay_exp,
            color="#d62728",
            alpha=0.15,
            label="Net Causal Gain",
        )
    ax2.set_title(
        r"B. Dynamic Exposure Timeline $E(t)$", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Simulation Step", fontsize=10)
    ax2.set_ylabel("Composite Exposure Metric", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    # 3. Panel C: Cold-Start (0 Baseline) vs. Existing Topic Lift
    ax3 = axes[0, 2]
    existing_runs = [
        r
        for r in runs
        if r.get("topic_mode") == "existing" and r.get("preset") in ("s1", "s3")
    ]
    cold_runs = [
        r
        for r in runs
        if r.get("topic_mode") == "cold_start" and r.get("preset") in ("s1", "s3")
    ]

    labels = ["S1 (Retrieval)", "S3 (Cascade)"]
    x = np.arange(len(labels))
    width = 0.35

    s1_exist = next((r["net_lift"] for r in existing_runs if r["preset"] == "s1"), 0.0)
    s3_exist = next((r["net_lift"] for r in existing_runs if r["preset"] == "s3"), 0.0)
    s1_cold = next((r["net_lift"] for r in cold_runs if r["preset"] == "s1"), 0.0)
    s3_cold = next((r["net_lift"] for r in cold_runs if r["preset"] == "s3"), 0.0)

    ax3.bar(
        x - width / 2,
        [s1_exist, s3_exist],
        width,
        label="Existing Topic",
        color="#2ca02c",
    )
    ax3.bar(
        x + width / 2,
        [s1_cold, s3_cold],
        width,
        label="Cold-Start (New)",
        color="#9467bd",
    )
    ax3.set_title(
        "C. Cold-Start vs. Existing Topic Net Lift", fontsize=12, fontweight="bold"
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel(r"Net Exposure Lift ($\Delta E$)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # 4. Panel D: Cross-Community Filter Bubble Penetration
    ax4 = axes[1, 0]
    multi_runs = [
        r
        for r in runs
        if r.get("topology") == "multitopic" and r.get("community_telemetry")
    ]
    if multi_runs:
        rep_run = multi_runs[-1]
        comm_stats = rep_run["community_telemetry"]
        comm_names = sorted(comm_stats.keys())
        impressions = [comm_stats[c].get("payload_impressions", 0) for c in comm_names]
        likes = [comm_stats[c].get("payload_likes", 0) for c in comm_names]
        idx_c = np.arange(len(comm_names))
        ax4.bar(
            idx_c - 0.15, impressions, 0.3, label="Feed Impressions", color="#17becf"
        )
        ax4.bar(idx_c + 0.15, likes, 0.3, label="Organic Likes", color="#e377c2")
        ax4.set_xticks(idx_c)
        ax4.set_xticklabels([c.capitalize() for c in comm_names])
    ax4.set_title(
        "D. Filter Bubble Penetration (Multi-Topic)", fontsize=12, fontweight="bold"
    )
    ax4.set_ylabel("Count", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()

    # 5. Panel E: Ideological Polarization Breakdown
    ax5 = axes[1, 1]
    pol_runs = [
        r
        for r in runs
        if r.get("topology") == "polarized" and r.get("community_telemetry")
    ]
    if pol_runs:
        rep_pol = pol_runs[-1]
        comm_p = rep_pol["community_telemetry"]
        p_names = sorted(comm_p.keys())
        p_likes = [comm_p[k].get("payload_likes", 0) for k in p_names]
        p_comm = [comm_p[k].get("payload_comments", 0) for k in p_names]
        idx_p = np.arange(len(p_names))
        ax5.bar(idx_p - 0.15, p_likes, 0.3, label="Likes", color="#3366cc")
        ax5.bar(idx_p + 0.15, p_comm, 0.3, label="Comments", color="#dc3912")
        ax5.set_xticks(idx_p)
        ax5.set_xticklabels([k.capitalize() for k in p_names])
    ax5.set_title(
        "E. Ideological Engagement (Polarized)", fontsize=12, fontweight="bold"
    )
    ax5.set_ylabel("Count", fontsize=10)
    ax5.grid(True, linestyle="--", alpha=0.6)
    ax5.legend()

    # 6. Panel F: Bot Fleet Efficiency Index
    ax6 = axes[1, 2]
    cib_runs = [r for r in runs if r.get("num_bots", 0) > 0]
    if cib_runs:
        run_labels = [
            f"{r['preset'].upper()} ({int(r.get('bot_ratio', 0) * 100)}%)"
            for r in cib_runs[:8]
        ]
        effs = [
            (r["exposure_payload"] - r["exposure_baseline"])
            / max(1, r.get("telemetry", {}).get("total_bot_actions", 1))
            for r in cib_runs[:8]
        ]
        ax6.barh(run_labels, effs, color="#bcbd22")
        ax6.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax6.set_title(
        r"F. Bot Cost-Efficiency Index ($\Delta E$ / Bot Actions)",
        fontsize=12,
        fontweight="bold",
    )
    ax6.set_xlabel("Net Exposure Gained per Action Expended", fontsize=10)
    ax6.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    logger.info(f"✓ Publication dashboard successfully saved to: {output_png}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate CIB batch experiment results and render publication dashboard."
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        required=True,
        help="Directory containing batch run folders / JSON files.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./cib_batch_summary.json",
        help="Path to save summary JSON.",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="./cib_batch_dashboard.png",
        help="Path to save 6-panel PNG dashboard.",
    )
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    logger.info(f"Scanning for batch experiment results in: {batch_dir}")

    runs = load_batch_runs(batch_dir)
    logger.info(f"Loaded {len(runs)} valid run results.")

    if not runs:
        logger.error(f"No valid run results found in {batch_dir}!")
        return 1

    summary = generate_summary_data(runs)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary table successfully saved to: {out_json}")

    out_png = Path(args.output_png)
    render_dashboard(runs, out_png)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
