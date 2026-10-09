import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default=os.getenv("DATABASE_URL", "postgresql+psycopg:///musictracker"), repr=False)
    request_timeout: float = 12.0
    request_interval: float = 0.4
    max_response_bytes: int = 2_000_000
    lease_seconds: int = 120
    planner_base_url: str = os.getenv("PLANNER_BASE_URL", "")
    planner_model: str = os.getenv("PLANNER_MODEL", "")
    planner_key: str = field(default=os.getenv("PLANNER_API_KEY", ""), repr=False)
    api_token: str = field(default=os.getenv("MUSICTRACKER_API_TOKEN", ""), repr=False)
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_key: str = field(default=os.getenv("DEEPSEEK_API_KEY", ""), repr=False)
    agent_max_rounds: int = 10
    agent_max_tools: int = 18
    agent_token_budget: int = 40000
    agent_timeout: float = 180.0
    model_timeout: float = 45.0


settings = Settings()
