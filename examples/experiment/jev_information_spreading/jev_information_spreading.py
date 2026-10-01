# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""JEV reproduction of OASIS 'Information Spreading in X' (paper F.2.2).

Mirrors the upstream Twitter harness
(examples/experiment/twitter_gpt_example/twitter_simulation.py) exactly in
platform / clock / dataset / agent-generation terms, but drives the simulation
with the high-performance JEV engine (`OasisEnv.step_jev`) backed by a vLLM
1-token logit classifier instead of per-agent CAMEL `perform_action_by_llm()`.

Paper baseline (F.2.2 "Align With Real Propagations"):
    * action space: {like_post, repost, follow, do_nothing}  (JEV: L, R, F, S)
    * 50 time steps, each = 3 minutes of sandbox time
    * compare first 150 simulated minutes against real propagation
    * ~300 agents (dataset-dependent)

T-08 note: this driver uses `step_jev` as the SOLE step driver. It never calls
`OasisEnv.step()`, so the sandbox clock advances strictly by +1 per step.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from yaml import safe_load

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from oasis.clock.clock import Clock
from oasis.environment.env import OasisEnv
from oasis.environment.jev_env import JEVExecutionConfig
from oasis.inference.jev_classifier import (
    MockJEVClassifierClient,
    VLLMJEVClassifierClient,
)
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

social_log = logging.getLogger(name="social.jev")
social_log.setLevel("INFO")
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(levelname)s %(asctime)s %(name)s %(message)s"))
social_log.addHandler(_h)

# Information Spreading baseline action set (paper Table 16, F.2.2).
BASELINE_ACTIONS: list[ActionType] = [
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
]

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/twitter_dataset/anonymous_topic_200_1h",
)
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "False_Business_0.csv")


def _build_classifier(inference_configs: dict[str, Any]):
    """Construct the JEV classifier client from inference config.

    backend == 'vllm'  -> VLLMJEVClassifierClient against inference.base_url
    backend == 'mock'  -> MockJEVClassifierClient (hermetic, no network)
    """
    backend = (inference_configs or {}).get("jev_backend", "vllm")
    if backend == "mock":
        social_log.info("JEV classifier backend: MOCK (hermetic)")
        return MockJEVClassifierClient()
    # Env wins over YAML: the HPC runner resolves the model PATH and the
    # isolated vLLM port at job time and exports them, so a stale YAML value
    # must not override the live server the sbatch just started.
    base_url = os.environ.get("VLLM_BASE_URL") or inference_configs.get(
        "base_url") or "http://127.0.0.1:8000/v1"
    model_name = os.environ.get("VLLM_MODEL") or inference_configs.get(
        "model_type") or "meta-llama/Meta-Llama-3-8B-Instruct"
    social_log.info(
        "JEV classifier backend: vLLM base_url=%s model=%s", base_url,
        model_name)
    return VLLMJEVClassifierClient(base_url=base_url, model_name=model_name)


async def running(
    db_path: str | None = ":memory:",
    csv_path: str | None = DEFAULT_CSV_PATH,
    num_timesteps: int = 50,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
    activation_prob: float | None = None,
    inference_configs: dict[str, Any] | None = None,
) -> None:
    inference_configs = inference_configs or {}
    db_path = ":memory:" if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    if db_path != ":memory:" and os.path.exists(db_path):
        os.remove(db_path)
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    start_time = 0  # Twitter branch uses integer sandbox time (upstream parity)
    clock = Clock(k=clock_factor)
    channel = Channel()
    infra = Platform(
        db_path=db_path,
        channel=channel,
        sandbox_clock=clock,
        start_time=start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )
    platform_task = asyncio.create_task(infra.running())

    # Agent LLM for profile generation / CAMEL plumbing (upstream parity).
    model = None
    model_type = inference_configs.get("model_type", "gpt-4o-mini")
    if str(model_type)[:3] == "gpt":
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType(model_type),
        )

    agent_graph = await generate_agents(
        agent_info_path=csv_path,
        channel=channel,
        start_time=start_time,
        model=model,
        recsys_type=recsys_type,
        available_actions=BASELINE_ACTIONS,
        twitter=infra,
    )

    # --- JEV engine wiring (the ONLY departure from the upstream harness) ---
    jev_cfg = JEVExecutionConfig(
        classifier_client=_build_classifier(inference_configs),
        max_actions_per_agent=1,   # one action per activated agent per step
        enable_belief_updates=False,  # Information Spreading has no stance dynamics
    )
    env = OasisEnv(
        agent_graph=agent_graph,
        platform=infra,
        database_path=None if db_path == ":memory:" else db_path,
    )

    # Paper F.2.2: 50 steps, 3 min/step. step_jev is the SOLE driver (T-08-safe).
    # The JEVExecutionConfig is consumed when the engine is lazily constructed on
    # the FIRST step_jev call; later calls reuse that engine, so config is passed
    # once (passing it again would be silently ignored by the OasisEnv wrapper).
    for timestep in range(1, num_timesteps + 1):
        clock.time_step = timestep * 3
        await infra.update_rec_table()
        step_kwargs: dict[str, Any] = dict(step_index=timestep, base_time=start_time)
        if timestep == 1:
            step_kwargs["config"] = jev_cfg
        result = await env.step_jev(**step_kwargs)
        counts = getattr(result, "action_counts", None) or getattr(
            result, "action_distribution", None)
        social_log.info("step %d/%d  actions=%s", timestep, num_timesteps,
                        counts)

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await platform_task
    social_log.info("JEV Information-Spreading simulation finished. DB=%s",
                    db_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JEV Information Spreading (OASIS F.2.2)")
    p.add_argument("--config_path", type=str, default="",
                   help="YAML config (data/simulation/inference sections).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.environ.setdefault("SANDBOX_TIME", "0")
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        asyncio.run(running(
            **(cfg.get("data") or {}),
            **(cfg.get("simulation") or {}),
            inference_configs=cfg.get("inference"),
        ))
    else:
        asyncio.run(running())
    social_log.info("Done.")
