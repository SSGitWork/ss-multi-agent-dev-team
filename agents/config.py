"""
Centralized configuration for the multi-agent system.

All settings are loaded from environment variables with sensible defaults.
No secrets are hardcoded - everything comes from .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AzureConfig:
    """Configuration for Azure OpenAI connectivity.

    Stores endpoint, deployment names, API version, and credentials
    required to access Azure-hosted OpenAI models.
    """
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_API_KEY", ""))
    api_base: str = field(default_factory=lambda: os.getenv("AZURE_API_BASE", os.getenv("AZURE_OPENAI_ENDPOINT", "")))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_API_VERSION", "2024-12-01-preview"))


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration used by each agent type.

    Defines which LLM deployment should be used by the Product Manager,
    Coder, and QA agents.
    """
    pm_deployment: str = field(default_factory=lambda: os.getenv("PM_MODEL_DEPLOYMENT", "gpt-4o"))
    pm_model: str = field(default_factory=lambda: os.getenv("PM_MODEL_NAME", "azure/gpt-4o"))
    coder_deployment: str = field(default_factory=lambda: os.getenv("CODER_MODEL_DEPLOYMENT", "gpt-4o-mini"))
    coder_model: str = field(default_factory=lambda: os.getenv("CODER_MODEL_NAME", "azure/gpt-4o-mini"))
    qa_deployment: str = field(default_factory=lambda: os.getenv("QA_MODEL_DEPLOYMENT", "gpt-4o-mini"))
    qa_model: str = field(default_factory=lambda: os.getenv("QA_MODEL_NAME", "azure/gpt-4o-mini"))


@dataclass(frozen=True)
class ResilienceConfig:
    """Configuration for retry and circuit breaker behavior.

    Controls retry counts, backoff delays, and circuit breaker thresholds
    used to protect LLM and tool calls from cascading failures.
    """
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3")))
    retry_base_delay: float = field(default_factory=lambda: float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0")))
    circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3")))
    circuit_breaker_cooldown: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_COOLDOWN", "60")))


@dataclass(frozen=True)
class CostConfig:
    """Configuration for token cost estimation.

    Defines per-token pricing used by the cost tracker to estimate
    pipeline execution cost.
    """
    gpt4o_input_per_1k: float = field(default_factory=lambda: float(os.getenv("GPT4O_INPUT_COST_PER_1K", "0.0025")))
    gpt4o_output_per_1k: float = field(default_factory=lambda: float(os.getenv("GPT4O_OUTPUT_COST_PER_1K", "0.01")))
    gpt4o_mini_input_per_1k: float = field(default_factory=lambda: float(os.getenv("GPT4O_MINI_INPUT_COST_PER_1K", "0.00015")))
    gpt4o_mini_output_per_1k: float = field(default_factory=lambda: float(os.getenv("GPT4O_MINI_OUTPUT_COST_PER_1K", "0.0006")))
    report_dir: str = field(default_factory=lambda: os.getenv("COST_REPORT_DIR", "./docs/cost_reports"))


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the prompt cache.

    Defines cache size and TTL used for caching LLM responses to reduce
    token usage and repeated model calls.
    """
    ttl: int = field(default_factory=lambda: int(os.getenv("PROMPT_CACHE_TTL", "3600")))
    max_size: int = field(default_factory=lambda: int(os.getenv("PROMPT_CACHE_MAX_SIZE", "100")))


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent execution behavior.

    Controls limits such as sliding window size, test execution timeout,
    and other runtime settings used by the agents.
    """
    max_react_iterations: int = field(default_factory=lambda: int(os.getenv("MAX_REACT_ITERATIONS", "10")))
    exec_timeout: int = field(default_factory=lambda: int(os.getenv("EXEC_TIMEOUT_SECONDS", "10")))
    sliding_window_size: int = field(default_factory=lambda: int(os.getenv("SLIDING_WINDOW_SIZE", "10")))
    max_qa_iterations: int = field(default_factory=lambda: int(os.getenv("MAX_QA_ITERATIONS", "5")))
    qa_test_timeout: int = field(default_factory=lambda: int(os.getenv("QA_TEST_TIMEOUT", "30")))


@dataclass(frozen=True)
class ChromaConfig:
    """Configuration for ChromaDB vector memory.

    Defines persistence directory, collection name, and connection mode
    for semantic memory storage.
    """
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "coder_agent_memory"))
    persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_store"))
    host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8000")))
    use_http: bool = field(default_factory=lambda: os.getenv("CHROMA_USE_HTTP", "false").lower() == "true")


@dataclass(frozen=True)
class PhoenixConfig:
    """Configuration for distributed tracing with Phoenix.

    Controls whether tracing is enabled and specifies the Phoenix endpoint
    and project name used for OpenTelemetry spans.
    """
    enabled: bool = field(default_factory=lambda: os.getenv("PHOENIX_ENABLED", "true").lower() == "true")
    project_name: str = field(default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "multi-agent-dev-team"))
    endpoint: str = field(default_factory=lambda: os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006"))


class Settings:
    """Singleton-like settings container."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.azure = AzureConfig()
            cls._instance.models = ModelConfig()
            cls._instance.resilience = ResilienceConfig()
            cls._instance.cost = CostConfig()
            cls._instance.cache = CacheConfig()
            cls._instance.agent = AgentConfig()
            cls._instance.chroma = ChromaConfig()
            cls._instance.phoenix = PhoenixConfig()
        return cls._instance


def get_settings() -> Settings:
    """Load and return the global application settings.

    Reads environment variables and returns a singleton configuration
    object containing all runtime settings used by the agents and
    orchestration pipeline.
    """
    return Settings()
