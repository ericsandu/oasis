# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from colorama import Back
from yaml import safe_load

scripts_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(scripts_dir)
from utils import create_model_urls

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

# JEV acceleration (opt-in). The shared helper decides classic vs batched JEV
# stepping; enable with OASIS_USE_JEV=1 or simulation.use_jev in the YAML.
from oasis.environment.env import OasisEnv
from oasis.environment.jev_runner import (
    jev_enabled,
    run_simulation_step,
    hourly_threshold_predicate,
)

social_log = logging.getLogger(name="social")
social_log.propagate = False
social_log.setLevel("DEBUG")

file_handler = logging.FileHandler("social.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/twitter_dataset/anonymous_topic_200_1h",
)
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "False_Business_0.csv")


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    csv_path: str | None = DEFAULT_CSV_PATH,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
    refresh_rec_post_count: int = 2,
    max_rec_post_len: int = 2,
    following_post_count: int = 3,
    model_configs: dict[str, Any] | None = None,
    inference_configs: dict[str, Any] | None = None,
    available_actions: list[ActionType] = None,
) -> None:
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    if os.path.exists(db_path):
        os.remove(db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if recsys_type == "reddit":
        start_time = datetime.now()
    else:
        start_time = 0
    social_log.info(f"Start time: {start_time}")
    clock = Clock(k=clock_factor)
    twitter_channel = Channel()
    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=refresh_rec_post_count,
        max_rec_post_len=max_rec_post_len,
        following_post_count=following_post_count,
    )
    twitter_task = asyncio.create_task(infra.running())
    # Prefer the runtime JEV_VLLM_URL (the sbatch picks a dynamic vLLM port) over
    # a static server_url in the YAML, so the config need not know the port.
    _env_url = os.environ.get("JEV_VLLM_URL")
    if _env_url:
        model_urls = [_env_url]
    else:
        model_urls = create_model_urls(inference_configs["server_url"])
    # The vLLM server is launched with --served-model-name set to the on-disk
    # path (e.g. /models/Meta-Llama-3-8B-Instruct), so requests MUST use that
    # exact name or vLLM returns 404 ("model does not exist") and every agent
    # LLM call silently fails -> agents only ever 'refresh'. Prefer the runtime
    # JEV_VLLM_MODEL (the sbatch exports the served name) over the YAML's HF
    # repo name, mirroring the JEV_VLLM_URL preference just above.
    _env_model = os.environ.get("JEV_VLLM_MODEL")
    _agent_model_type = _env_model or inference_configs["model_type"]
    if _env_model:
        social_log.info("CLASSIC agent model_type <- JEV_VLLM_MODEL=%r "
                        "(served name; YAML model_type ignored)", _env_model)
    # --- Classic baseline tool-calling fix ---------------------------------
    # Plain Meta-Llama-3-8B-Instruct narrates action intent in prose and emits
    # EMPTY tool_calls under vLLM's auto parser (its tokenizer lacks the
    # <|python_tag|> token the llama3_json parser needs). Forcing
    # tool_choice="required" makes vLLM use STRUCTURED/GUIDED decoding to emit a
    # schema-valid tool call every turn (supported since vLLM>=0.8.3; needs NO
    # --tool-call-parser), capturing the intent base OASIS otherwise drops.
    # Applied to the CLASSIC path ONLY: JEV bypasses tool-calling entirely via
    # its own logit-bias classifier, so this must not touch the JEV run.
    # Override/disable with OASIS_CLASSIC_TOOL_CHOICE (e.g. "auto"/"none"/"").
    _classic_tool_choice = os.environ.get("OASIS_CLASSIC_TOOL_CHOICE", "required")
    _classic_is_jev = jev_enabled((inference_configs or {}).get("use_jev"))
    _agent_model_config: dict[str, Any] = {}
    if (not _classic_is_jev) and _classic_tool_choice:
        _agent_model_config["tool_choice"] = _classic_tool_choice
        social_log.info("CLASSIC agent model: tool_choice=%r (forced structured "
                        "tool emission)", _classic_tool_choice)
    models = [
        ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type=_agent_model_type,
            url=url,
            model_config_dict=_agent_model_config or None,
        ) for url in model_urls
    ]
    try:
        all_topic_df = pd.read_csv("data/twitter_dataset/all_topics.csv")
        if "False" in csv_path or "True" in csv_path:
            if "-" not in csv_path:
                topic_name = csv_path.split("/")[-1].split(".")[0]
            else:
                topic_name = csv_path.split("/")[-1].split(".")[0].split(
                    "-")[0]
            source_post_time = (
                all_topic_df[all_topic_df["topic_name"] ==
                             topic_name]["start_time"].item().split(" ")[1])
            start_hour = int(source_post_time.split(":")[0]) + float(
                int(source_post_time.split(":")[1]) / 60)
    except Exception:
        print("No real-world data, let start_hour be 1PM")
        start_hour = 13

    model_configs = model_configs or {}

    agent_graph = await generate_agents(agent_info_path=csv_path,
                                        channel=twitter_channel,
                                        start_time=start_time,
                                        model=models,
                                        recsys_type=recsys_type,
                                        available_actions=available_actions,
                                        twitter=infra)
    # agent_graph.visualize("initial_social_graph.png")

    # --- JEV acceleration setup (opt-in; classic path untouched when off) ---
    _use_jev = jev_enabled((inference_configs or {}).get("use_jev"))
    _jev_env = None
    _jev_cfg = None
    if _use_jev:
        from oasis.environment.jev_env import JEVExecutionConfig
        from oasis.inference.jev_classifier import (
            DEFAULT_ACTION_TOKEN_MAP, VLLMJEVClassifierClient)
        # Build the classifier against the same vLLM server, restricted to the
        # action set OASIS was given for this scenario (available_actions).
        # available_actions entries may be ActionType OR action-name strings
        # (the yaml_200 configs use strings like "like_post"); handle both.
        char_by_name = {
            "like_post": "L", "repost": "R", "quote_post": "Q",
            "create_comment": "C", "follow": "F", "do_nothing": "S",
        }

        def _action_char(a):
            name = getattr(a, "value", a)  # ActionType.value or raw string
            return char_by_name.get(str(name))

        allowed_chars = [c for c in (_action_char(a)
                         for a in (available_actions or [])) if c] \
            or ["L", "R", "F", "S"]
        if "S" not in allowed_chars:
            allowed_chars.append("S")
        logit_bias = {}
        for ch in allowed_chars:
            for tid in DEFAULT_ACTION_TOKEN_MAP.get(ch, []):
                logit_bias[tid] = 50.0
        token_id_map = {ch: DEFAULT_ACTION_TOKEN_MAP[ch][0]
                        for ch in allowed_chars if ch in DEFAULT_ACTION_TOKEN_MAP}
        jev_url = os.environ.get("JEV_VLLM_URL") or (
            inference_configs or {}).get("base_url", "http://127.0.0.1:8000/v1")
        jev_model = os.environ.get("JEV_VLLM_MODEL") or (
            inference_configs or {}).get("model_type", "")
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        # Match base-OASIS classic sampling: classic agents go through CAMEL's
        # VLLMConfig, whose temperature defaults to None -> vLLM server default
        # 1.0 (full sampling). JEV's classifier defaulted to temperature=0.0
        # (GREEDY argmax), which deterministically picks the single top action
        # (always Like for an isolated post) and NEVER samples the lower-but-
        # real-probability Repost -- the true cause of JEV's 0-repost collapse,
        # not the prompt. Match classic by sampling at 1.0. Override via
        # OASIS_JEV_CLASSIFY_TEMP.
        _jev_temp = float(os.environ.get("OASIS_JEV_CLASSIFY_TEMP", "1.0"))
        social_log.info("JEV ENABLED: actions=%s url=%s classify_temp=%.2f",
                        allowed_chars, jev_url, _jev_temp)
        _jev_cfg = JEVExecutionConfig(
            classifier_client=VLLMJEVClassifierClient(
                base_url=jev_url, model_name=jev_model,
                logit_bias=logit_bias, token_id_map=token_id_map,
                temperature=_jev_temp),
            max_actions_per_agent=1,
            enable_belief_updates=False,
            # Enabled action letters (from the config's available_actions) so the
            # JEV task-instruction prompt lists ONLY these actions, matching what
            # a classic agent would be given -- not a hardcoded menu.
            allowed_actions=allowed_chars,
            # WHOLE-FEED experiment toggle (OASIS_JEV_FEED_MODE=1): one call per
            # agent over the full feed instead of per-post. Default off.
            feed_mode=(os.environ.get("OASIS_JEV_FEED_MODE", "0") == "1"),
            # The driver already calls infra.update_rec_table() once per step
            # (the expensive twhin-BERT embedding pass). Don't let step_jev run
            # it a SECOND time -- that doubled the heaviest op every step.
            update_recsys=False,
        )
        _jev_env = OasisEnv(agent_graph=agent_graph, platform=infra,
                            database_path=db_path)

    # Sim-only wall clock: brackets ONLY the step loop (steps 1..N), excluding
    # imports, recsys-model load, and agent-graph generation -- the fixed
    # startup overhead that otherwise inflates a per-process timer and masks the
    # true classic-vs-JEV difference. Printed as a parseable line for the
    # benchmark harness.
    import time as _time
    _sim_t0 = _time.perf_counter()
    for timestep in range(1, num_timesteps + 1):
        clock.time_step = timestep * 3
        social_log.info(f"timestep:{timestep}")
        db_file = db_path.split("/")[-1]
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        # if you want to disable recsys, please comment this line
        await infra.update_rec_table()

        # 0.05 * timestep here means 3 minutes / timestep
        simulation_time_hour = start_hour + 0.05 * timestep

        if not _use_jev:
            # ---- VERBATIM UPSTREAM CLASSIC PATH (base-OASIS reference) ----
            # Byte-for-byte the original OASIS loop, incl. the hourly
            # active_threshold activation (now populated by generate_agents).
            tasks = []
            for node_id, agent in agent_graph.get_agents():
                if agent.user_info.is_controllable is False:
                    agent_ac_prob = random.random()
                    threshold = agent.user_info.profile["other_info"][
                        "active_threshold"][int(simulation_time_hour % 24)]
                    if agent_ac_prob < threshold:
                        tasks.append(agent.perform_action_by_llm())
                else:
                    await agent.perform_action_by_hci()
            await asyncio.gather(*tasks)
        else:
            # ---- JEV path: batched env.step_jev() over the SAME threshold-
            # activated agents, so JEV and classic are comparable. ----
            await run_simulation_step(
                env=_jev_env,
                agent_graph=agent_graph,
                step_index=timestep,
                base_time=start_time,
                use_jev=True,
                active_predicate=hourly_threshold_predicate(simulation_time_hour),
                jev_kwargs=({"config": _jev_cfg} if timestep == 1 else None),
            )

    _sim_wall = _time.perf_counter() - _sim_t0
    # Parseable marker for the benchmark harness (sim-only, excludes startup).
    print(f"SIM_WALL_SECONDS={_sim_wall:.3f} engine={'jev' if _use_jev else 'classic'}")
    social_log.info("SIM_WALL_SECONDS=%.3f engine=%s", _sim_wall,
                    "jev" if _use_jev else "classic")
    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await twitter_task


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation") or {}
        inference_configs = cfg.get("inference") or {}

        # use_jev may be placed under simulation (natural) or inference; keep it
        # OUT of running()'s kwargs (it is read from inference_configs inside).
        if "use_jev" in simulation_params:
            inference_configs.setdefault(
                "use_jev", simulation_params.pop("use_jev"))

        asyncio.run(
            running(**data_params,
                    **simulation_params,
                    inference_configs=inference_configs))
    else:
        asyncio.run(running())
    social_log.info("Simulation finished.")
