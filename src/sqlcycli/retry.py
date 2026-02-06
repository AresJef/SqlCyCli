# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.set import PySet_Contains as set_contains  # type: ignore

# Python imports
import warnings
import functools, inspect, asyncio, time
from typing import Callable, Awaitable, Any, ParamSpec, TypeVar
from sqlcycli import errors


P = ParamSpec("P")
R = TypeVar("R")


def retry_on_errno(
    errno_codes: tuple[int, ...],
    retry_attempts: cython.int = 3,
    retry_wait_time: cython.double = 1.0,
) -> Callable[[Callable[P, R] | Callable[P, Awaitable[R]]], Callable[P, Any]]:
    """Decorator that retries sync/async functions when raised `<'MySQLError'>` with errno matches `errno_codes`.

    :param errno_codes `<'tuple[int]'>`: Tuple of errno code(s) from `<'MySQLError'>` that should trigger retry.
    :param retry_attempts `<'int'>`: Total attempts (does not include the first call). Defaults to `3`.
        If `retry_attempts<=0`, it means infinite retries until success or non-retryable error.
    :param retry_wait_time `<'float'>`: Seconds to wait between retries. Defaults to `1.0`.
        Automatically clamped to `0.0` if negative.
    """
    if len(errno_codes) == 0:
        raise errors.RetryValueError("errno_codes cannot be empty.")
    errno_set: set = set(errno_codes)
    if retry_attempts < 0:
        retry_attempts = 0  # infinite retries
    if retry_wait_time < 0.0:
        retry_wait_time = 0.0

    def decorator(func: Callable[P, R] | Callable[P, Awaitable[R]]) -> Callable[P, Any]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> R:
            attempts: cython.int = 0
            while True:
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc]
                except errors.MySQLError as exc:
                    if retry_attempts > 0 and attempts >= retry_attempts:
                        raise
                    if not set_contains(errno_set, exc.errno):
                        raise
                    attempts += 1
                    warnings.warn(
                        "Retrying (%d) on failed query (%s)" % (attempts, exc),
                        stacklevel=2,
                    )
                    if retry_wait_time > 0:
                        await asyncio.sleep(retry_wait_time)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> R:
            attempts: cython.int = 0
            while True:
                try:
                    return func(*args, **kwargs)  # type: ignore[misc]
                except errors.MySQLError as exc:
                    if retry_attempts > 0 and attempts >= retry_attempts:
                        raise
                    if not set_contains(errno_set, exc.errno):
                        raise
                    attempts += 1
                    warnings.warn(
                        "Retrying (%d) on failed query (%s)" % (attempts, exc),
                        stacklevel=2,
                    )
                    if retry_wait_time > 0:
                        time.sleep(retry_wait_time)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
