"""Bounded retries shared by the source API and AI extraction.

No upstream body, request URL, headers or exception string is logged: providers
may echo credentials. Only fixed, actionable error descriptions leave this layer.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time
from typing import Callable, TypeVar

import requests

T = TypeVar("T")
MAX_ATTEMPTS = 3
MAX_WAIT_SECONDS = 30


class ServiceError(Exception):
    pass


class PermanentServiceError(ServiceError):
    """Authentication/configuration error: stop the batch as well as this request."""


class TemporaryServiceError(ServiceError):
    """A service outage/timeout exhausted its budget: pause the rest of the batch."""


class RetryableError(ServiceError):
    def __init__(self, message: str, retry_after: float | None = None, *, stop_batch: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.stop_batch = stop_batch


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def with_retries(operation: Callable[[int], T], *, sleep=time.sleep) -> T:
    for attempt in range(MAX_ATTEMPTS):
        try:
            return operation(attempt)
        except RetryableError as exc:
            delay = exc.retry_after if exc.retry_after is not None else 2 ** (attempt + 1)
            if delay > MAX_WAIT_SECONDS:
                raise TemporaryServiceError("服务要求较长等待，本轮已暂停；请稍后重试。") from None
            if attempt == MAX_ATTEMPTS - 1:
                error_type = TemporaryServiceError if exc.stop_batch else ServiceError
                raise error_type(f"{exc}（已达 {MAX_ATTEMPTS} 次尝试上限）") from None
            sleep(delay)
    raise AssertionError("unreachable")


def request_once(session, method: str, url: str, *, service: str, **kwargs):
    try:
        response = session.request(method, url, allow_redirects=False, **kwargs)
    except requests.Timeout:
        raise RetryableError(f"{service}请求超时", stop_batch=True) from None
    except requests.RequestException:
        raise RetryableError(f"{service}网络连接失败，请检查网络或代理", stop_batch=True) from None
    status = response.status_code
    if status == 429:
        delay = retry_after_seconds(response.headers.get("Retry-After"))
        response.close()
        raise RetryableError(f"{service}限流（HTTP 429）", delay, stop_batch=True)
    if status in {408, 425, 500, 502, 503, 504}:
        response.close()
        reason = "服务繁忙（HTTP 503），不代表密钥错误，请稍后重试" if status == 503 else f"服务暂时不可用（HTTP {status}）"
        raise RetryableError(f"{service}{reason}", stop_batch=True)
    if status in {400, 401, 402, 403, 404, 422}:
        reasons = {400: "请求参数或模型配置不正确", 401: "密钥无效", 402: "账户余额不足",
                   403: "访问权限不足", 404: "接口地址或模型不存在", 422: "请求参数不兼容"}
        response.close()
        raise PermanentServiceError(f"{service}：{reasons[status]}（HTTP {status}），请修正后重试。")
    if not 200 <= status < 300:
        response.close()
        raise ServiceError(f"{service}返回 HTTP {status}，本轮停止。")
    return response
