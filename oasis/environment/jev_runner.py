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
"""Shared JEV stepping hook (Option B).

ONE place that decides, per simulation step, between:
  * classic OASIS stepping -- per-agent ``agent.perform_action_by_llm()`` fanned
    out with ``asyncio.gather`` (unchanged upstream behavior), and
  * batched JEV stepping -- a single ``env.step_jev()`` call over the whole
    active population (1-token logit classification + micro-time scheduling).

Every scenario driver (twitter_simulation_large, group_polar, reddit_*,
1M_agents, ...) performs the SAME per-step loop:

    for _, agent in agent_graph.get_agents():
        if active(agent):
            tasks.append(agent.perform_action_by_llm())
    await asyncio.gather(*tasks)

Replacing that block with a single call to :func:`run_simulation_step` lets any
driver gain JEV acceleration by flipping one flag -- the JEV switch is written
HERE, once, not copied into each driver.

Enable JEV by env var (OASIS_USE_JEV=1), by passing use_jev=True, or via a
scenario YAML's ``simulation.use_jev`` which the driver forwards here.
"""
from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Awaitable, Callable, Optional


def jev_enabled(explicit: Optional[bool] = None) -> bool:
    """Resolve whether JEV stepping is on.

    Precedence: explicit arg > OASIS_USE_JEV env var > False.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("OASIS_USE_JEV", "0").lower() in ("1", "true", "yes")


async def run_simulation_step(
    *,
    env: Any,
    agent_graph: Any,
    step_index: int,
    base_time: Any = None,
    use_jev: Optional[bool] = None,
    active_predicate: Optional[Callable[[Any], bool]] = None,
    jev_kwargs: Optional[dict[str, Any]] = None,
    classic_action: str = "perform_action_by_llm",
) -> Any:
    """Execute one simulation step, classic or JEV, over the active agents.

    Args:
        env: the OasisEnv (must expose ``step_jev`` for the JEV path).
        agent_graph: provides ``get_agents()`` -> iterable of (node_id, agent).
        step_index: integer step index (passed to step_jev; also used by caller).
        base_time: base simulation time handed to step_jev.
        use_jev: tri-state override; falls back to OASIS_USE_JEV env.
        active_predicate: fn(agent) -> bool deciding if an agent acts this step.
            Defaults to the standard OASIS hourly-threshold activation when the
            agent exposes it, else "always active for non-controllable".
        jev_kwargs: extra kwargs forwarded to env.step_jev on the JEV path
            (e.g. {"config": JEVExecutionConfig(...)} on the FIRST step only).
        classic_action: the per-agent coroutine method name for the classic path.

    Returns:
        The JEVStepResult on the JEV path, else None (classic path dispatches
        via the agents directly, like upstream).
    """
    agents = list(agent_graph.get_agents())

    def _is_active(agent: Any) -> bool:
        if active_predicate is not None:
            return active_predicate(agent)
        # Default: controllable agents are driven elsewhere (HCI); a
        # non-controllable agent acts every step unless the caller supplied a
        # richer predicate (the drivers pass the hourly-threshold one).
        return getattr(getattr(agent, "user_info", None),
                       "is_controllable", False) is False

    if jev_enabled(use_jev):
        # --- Batched JEV path: one classification pass over active agents. ---
        active_ids = []
        for _node_id, agent in agents:
            if _is_active(agent):
                aid = getattr(agent, "social_agent_id",
                              getattr(agent, "agent_id", None))
                if aid is not None:
                    active_ids.append(int(aid))
        return await env.step_jev(
            step_index=step_index,
            base_time=base_time,
            active_agent_ids=active_ids or None,
            **(jev_kwargs or {}),
        )

    # --- Classic OASIS path: per-agent LLM, unchanged upstream behavior. ---
    tasks: list[Awaitable[Any]] = []
    for _node_id, agent in agents:
        if getattr(getattr(agent, "user_info", None),
                   "is_controllable", False):
            # Controllable agents act via HCI, as upstream.
            hci = getattr(agent, "perform_action_by_hci", None)
            if hci is not None:
                await hci()
            continue
        if _is_active(agent):
            method = getattr(agent, classic_action)
            tasks.append(method())
    if tasks:
        await asyncio.gather(*tasks)
    return None


def hourly_threshold_predicate(simulation_time_hour: float) -> Callable[[Any], bool]:
    """Build the standard OASIS activation predicate for a given sim hour.

    Mirrors the per-driver block:
        thr = profile['other_info']['active_threshold'][hour % 24]
        active iff random() < thr
    Controllable agents are excluded (driven via HCI).
    """
    hour_idx = int(simulation_time_hour % 24)

    def _pred(agent: Any) -> bool:
        info = getattr(agent, "user_info", None)
        if info is None or getattr(info, "is_controllable", False):
            return False
        try:
            thr = info.profile["other_info"]["active_threshold"][hour_idx]
        except (KeyError, IndexError, TypeError, AttributeError):
            thr = 1.0  # no threshold -> always active (upstream fallback)
        return random.random() < thr

    return _pred
