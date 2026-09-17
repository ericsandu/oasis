# =========== Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. ===========
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
# =========== Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import urllib.request
import json
from typing import List, Optional, Union

from camel.models import ModelFactory, ModelManager
from camel.types import ModelPlatformType
from camel.configs import VLLMConfig


DEFAULT_VLLM_MODEL = os.environ.get(
    "VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
)
DEFAULT_VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")


def check_vllm_health(url: Optional[str] = None, timeout: float = 3.0) -> bool:
    """Check if the local or remote vLLM server is healthy.
    
    Args:
        url: URL to the health or v1 endpoint. If omitted, uses DEFAULT_VLLM_URL.
        timeout: Timeout in seconds for the health check.
        
    Returns:
        True if the server responded with 200 OK, False otherwise.
    """
    base_url = url or DEFAULT_VLLM_URL
    if base_url.endswith("/v1"):
        health_url = base_url[:-3] + "/health"
    elif base_url.endswith("/v1/"):
        health_url = base_url[:-4] + "/health"
    else:
        health_url = f"{base_url.rstrip('/')}/health"

    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "OASIS-HealthCheck"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        try:
            models_url = f"{base_url.rstrip('/')}/models" if not base_url.endswith("/models") else base_url
            req = urllib.request.Request(models_url, headers={"User-Agent": "OASIS-HealthCheck"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False


def resolve_registered_vllm_model(url: str, requested_model: str) -> str:
    """Query vLLM /v1/models to verify or auto-resolve the exact registered model name."""
    models_url = url.rstrip("/")
    if not models_url.endswith("/models"):
        if models_url.endswith("/v1"):
            models_url = f"{models_url}/models"
        else:
            models_url = f"{models_url}/v1/models"
    try:
        req = urllib.request.Request(models_url, headers={"User-Agent": "OASIS-ModelResolver"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
            available_models = [m["id"] for m in data.get("data", [])]
            if requested_model in available_models:
                return requested_model
            # Check for basename match (e.g. /models/Qwen3.8-27B vs Qwen3.8-27B)
            req_base = os.path.basename(requested_model.rstrip("/"))
            for m in available_models:
                if m == req_base or os.path.basename(m.rstrip("/")) == req_base:
                    print(f"✓ Auto-resolved model '{requested_model}' to registered vLLM name '{m}'")
                    return m
            if available_models:
                print(f"✓ Using first available vLLM model: '{available_models[0]}' (requested: '{requested_model}')")
                return available_models[0]
    except Exception as e:
        pass
    return requested_model


def get_vllm_model(
    model_type: Optional[str] = None,
    url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.95,
):
    """Create a CAMEL model instance connected to the central vLLM node.
    
    Args:
        model_type: HuggingFace model identifier (defaults to Qwen-2.5-32B-Instruct-GPTQ-Int8).
        url: vLLM OpenAI-compatible endpoint (defaults to http://127.0.0.1:8000/v1).
        temperature: Sampling temperature for agent generation.
        max_tokens: Maximum tokens to generate per agent turn.
        top_p: Nucleus sampling top_p.
        
    Returns:
        CAMEL ModelBackend instance backed by VLLM.
    """
    server_url = url or DEFAULT_VLLM_URL
    raw_model_name = model_type or DEFAULT_VLLM_MODEL
    model_name = resolve_registered_vllm_model(server_url, raw_model_name)

    config = VLLMConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type=model_name,
        url=server_url,
        model_config_dict=config.as_dict(),
    )

    # Sanitize messages sent to vLLM so only index 0 can be role='system'
    def _sanitize_messages(msgs):
        if not msgs:
            return msgs
        clean = []
        for idx, m in enumerate(msgs):
            if isinstance(m, dict) and m.get("role") == "system" and idx > 0:
                cm = dict(m)
                cm["role"] = "user"
                clean.append(cm)
            else:
                clean.append(m)
        return clean

    orig_arun = model.arun

    async def safe_arun(messages, *args, **kwargs):
        return await orig_arun(_sanitize_messages(messages), *args, **kwargs)

    model.arun = safe_arun

    if hasattr(model, "run"):
        orig_run = model.run

        def safe_run(messages, *args, **kwargs):
            return orig_run(_sanitize_messages(messages), *args, **kwargs)

        model.run = safe_run

    return model


def get_vllm_model_manager(
    urls: Optional[List[str]] = None,
    model_type: Optional[str] = None,
    scheduling_strategy: str = "round_robin",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> ModelManager:
    """Create a ModelManager load-balancing across one or more vLLM endpoints.
    
    Args:
        urls: List of vLLM /v1 URLs (e.g. ['http://127.0.0.1:8000/v1', 'http://127.0.0.1:8001/v1']).
        model_type: Model identifier.
        scheduling_strategy: Scheduling strategy (e.g. 'round_robin').
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        
    Returns:
        ModelManager instance distributing requests across the endpoints.
    """
    endpoint_urls = urls or [DEFAULT_VLLM_URL]
    models = [
        get_vllm_model(
            model_type=model_type,
            url=u,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for u in endpoint_urls
    ]
    return ModelManager(models=models, scheduling_strategy=scheduling_strategy)
