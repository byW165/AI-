"""Load local configuration without exposing the API key in repr or logs."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent


class ConfigurationError(Exception):
    pass


@dataclass(frozen=True)
class AIConfig:
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)

    def validate(self) -> None:
        if not self.base_url or not self.model or not self.api_key:
            raise ConfigurationError("AI 尚未配置完整：请在项目 .env 填写 API 地址、模型名和密钥，保存后重新点击操作。")
        url = urlsplit(self.base_url)
        if (url.scheme not in {"https", "http"} or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or self.base_url.endswith("/chat/completions")):
            raise ConfigurationError("AI_BASE_URL 应为基础地址，不含密钥、查询参数或 /chat/completions。")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ConfigurationError("远程 AI 接口请使用 HTTPS 地址。")


def load_config() -> AIConfig:
    # Read on every action so saving .env takes effect without a server restart.
    local = dotenv_values(ROOT / ".env", interpolate=False)
    def read(name: str) -> str:
        return str(os.environ.get(name, local.get(name)) or "").strip()
    return AIConfig(read("AI_BASE_URL").rstrip("/"), read("AI_MODEL"), read("AI_API_KEY"))
