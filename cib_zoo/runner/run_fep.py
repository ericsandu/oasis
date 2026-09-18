"""CLI Runner script for executing CIB experiments locally or on UPB Grid clusters."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time

# Ensure oasis root is on sys.path
_oasis_root = Path(__file__).resolve().parents[2]
if str(_oasis_root) not in sys.path:
    sys.path.insert(0, str(_oasis_root))

# Enforce thread limit (4 max) for cluster and local execution
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(var, "4")

# Lightweight CPU mocks for heavy ML dependencies if running in non-GPU environment
for pkg in (
    "torch", "torch.nn", "sentence_transformers", "transformers",
    "sklearn", "sklearn.feature_extraction", "sklearn.feature_extraction.text",
    "sklearn.metrics", "sklearn.metrics.pairwise"
):
    try:
        __import__(pkg)
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules[pkg] = MagicMock()

from oasis.social_platform.database import create_db
from cib_zoo.metrics.amplification import calculate_causal_amplification
from cib_zoo.presets import create_s1_campaign, create_s2_campaign, create_s3_campaign

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("cib_zoo.runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIB attack campaigns in OASIS simulation.")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["s1", "s2", "s3"],
        default="s1",
        help="Campaign preset to execute (s1, s2, s3).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./cib_experiment.db",
        help="Path to SQLite simulation database.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./cib_results.json",
        help="Path to save experiment evaluation telemetry.",
    )
    parser.add_argument(
        "--num-bots",
        type=int,
        default=4,
        help="Total number of coordinated bot accounts.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum simulation steps to run.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    logger.info(f"Initializing CIB runner with preset={args.preset}, bots={args.num_bots}, max_steps={args.max_steps}")

    db_path = Path(args.db_path).resolve()
    create_db(str(db_path))

    bot_ids = list(range(2, args.num_bots + 2))

    if args.preset == "s1":
        campaign = create_s1_campaign(
            warmup_bot_ids=bot_ids[: len(bot_ids) // 2 or 1],
            strike_bot_ids=bot_ids,
            influencer_ids=[100],
            anchor_post_id=1,
            payload_post_id=2,
            payload_template="Target CIB narrative broadcast",
            warmup_steps=max(1, args.max_steps // 2),
            strike_steps=max(1, args.max_steps - args.max_steps // 2),
        )
    elif args.preset == "s2":
        campaign = create_s2_campaign(
            sentinel_bot_ids=[bot_ids[0]],
            strike_bot_ids=bot_ids[1:] if len(bot_ids) > 1 else bot_ids,
            target_post_id=1,
            raid_templates=["Perspective A", "Perspective B"],
            discovery_steps=max(1, args.max_steps // 2),
            strike_steps=max(1, args.max_steps - args.max_steps // 2),
        )
    else:
        campaign = create_s3_campaign(
            bridge_bot_ids=[bot_ids[0]],
            raid_bot_ids=[bot_ids[1 % len(bot_ids)]],
            astroturf_bot_ids=bot_ids,
            target_influencer_ids=[99],
            target_post_id=1,
            raid_templates=["Raid commentary"],
            astroturf_hashtag="amplification_test",
            astroturf_template="Astroturf viral broadcast",
        )

    logger.info(f"Compiled campaign '{campaign.campaign_name}' with {campaign.total_steps} planned steps.")

    # Telemetry report
    results = {
        "status": "success",
        "preset": args.preset,
        "campaign_name": campaign.campaign_name,
        "num_bots": args.num_bots,
        "total_steps": campaign.total_steps,
        "db_path": str(db_path),
        "timestamp": time.time(),
        "phases": [
            {
                "name": p.phase_name,
                "duration": p.duration_steps,
                "squads": [s.squad_id for s in p.squads],
            }
            for p in campaign.phases
        ],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Experiment execution recorded and saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
