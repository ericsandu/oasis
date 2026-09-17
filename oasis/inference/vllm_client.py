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
    model_name = model_type or DEFAULT_VLLM_MODEL
    server_url = url or DEFAULT_VLLM_URL

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
