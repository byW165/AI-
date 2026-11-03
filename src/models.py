"""Public data contracts; credentials never belong in these models."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("时间必须包含时区")
    return result.astimezone(timezone.utc)


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    abstract: str
    url: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            (self.title + "\n" + self.abstract + "\n" + self.updated).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class Summary:
    core_contribution: str
    methods_and_datasets: str
    code_status: str
