# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.set import PySet_Contains as set_contains  # type: ignore

# Python imports
import logging
from typing_extensions import Self
import functools, inspect, asyncio, time
from typing import Callable, Awaitable, Any, ParamSpec, TypeVar
from sqlcycli import errors

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# Retry on errno -----------------------------------------------------------------------------------------------
def retry_on_errno(
    errno_codes: tuple[int, ...],
    retry_attempts: cython.int = 3,
    retry_wait_time: cython.double = 1.0,
) -> Callable[[Callable[P, R] | Callable[P, Awaitable[R]]], Callable[P, Any]]:
    """Decorator that retries sync/async callables on `errors.MySQLError` when `exc.errno` matches `errno_codes`.

    This decorator supports both synchronous and asynchronous functions:
    - For sync callables, retry wait uses `time.sleep(...)`.
    - For async callables, retry wait uses `await asyncio.sleep(...)`.

    Retry policy:
    - Only `errors.MySQLError` is handled.
    - A retry is triggered only when `exc.errno` is contained in `errno_codes`.
    - If errno does not match, the original exception is raised immediately.

    Retry counting semantics:
    - `retry_attempts` is the number of retries **after** the initial call.
    - Total max executions = `1 + retry_attempts` when `retry_attempts > 0`.
    - If `retry_attempts <= 0`, retries are treated as infinite.

    Wait time semantics:
    - `retry_wait_time` is clamped to `0.0` when negative.
    - If `retry_wait_time == 0.0`, retries occur immediately (no sleep).

    Logging:
    - Each retry emits a warning with exception text, retry count, matched errno,
    and wait duration.

    :param errno_codes `<'tuple[int]'>`: MySQL errno values that should trigger retry.
        Must not be empty.
    :param retry_attempts `<'int'>`: Number of retries after the first attempt. Defaults to `3`.
        If `retry_attempts<=0`, retry indefinitely until success or non-retryable error.
    :param retry_wait_time `<'float'>`: Seconds to wait between retries. Defaults to `1.0`.
        Negative values are clamped to `0.0`.
    :return `<'Callable'>`: A decorator that wraps a sync/async callable with errno-based retry logic.
    :raises errors.RetryValueError: If `errno_codes` is empty.

    Example
    -------
    >>> @retry_on_errno((2003, 2013, 1205), retry_attempts=5, retry_wait_time=0.5)
    ... def run_query() -> None:
    ...     ...
    """
    if len(errno_codes) == 0:
        raise errors.RetryValueError("errno_codes cannot be empty.")
    errno_set: set = set(errno_codes)
    retry_attempts = max(retry_attempts, 0)  # clamp to 0
    retry_wait_time = max(retry_wait_time, 0.0)  # clamp to 0.0

    def decorator(func: Callable[P, R] | Callable[P, Awaitable[R]]) -> Callable[P, Any]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> R:
            retries: cython.longlong = 0
            while True:
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc]
                except errors.MySQLError as exc:
                    if retry_attempts > 0 and retries >= retry_attempts:
                        raise
                    errno = exc.errno
                    if not set_contains(errno_set, errno):
                        raise
                    retries += 1
                    # fmt: off
                    logger.warning(
                        "%s, retry (%d) on errno [%s] in %.1f seconds",
                        exc, retries, errno, retry_wait_time,
                    )
                    # fmt: on
                    if retry_wait_time > 0:
                        await asyncio.sleep(retry_wait_time)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> R:
            retries: cython.longlong = 0
            while True:
                try:
                    return func(*args, **kwargs)  # type: ignore[misc]
                except errors.MySQLError as exc:
                    if retry_attempts > 0 and retries >= retry_attempts:
                        raise
                    errno = exc.errno
                    if not set_contains(errno_set, errno):
                        raise
                    retries += 1
                    # fmt: off
                    logger.warning(
                        "%s, retry (%d) on errno [%s] in %.1f seconds",
                        exc, retries, errno, retry_wait_time,
                    )
                    # fmt: on
                    if retry_wait_time > 0:
                        time.sleep(retry_wait_time)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@cython.cclass
class RetryOnErrno:
    """Retry a sync or async block when `errors.MySQLError.errno` matches.

    Supports both sync and async usage.

    Sync usage
    ----------
    ```python
    for attempt in RetryOnErrno(
        (2003, 2013, 1205),
        retry_attempts=3,
        retry_wait_time=1.0,
    ):
        with attempt:
            # Some SQL execution
    ```

    Async usage
    -----------
    ```python
    async for attempt in RetryOnErrno(
        (2003, 2013, 1205),
        retry_attempts=3,
        retry_wait_time=1.0,
    ):
        async with attempt:
            # Some SQL execution
    ```
    """

    _errno_set: set[int]
    _retry_attempts: cython.int
    _retry_wait_time: cython.double
    _retries: cython.longlong
    _should_retry: cython.bint

    def __init__(
        self,
        errno_codes: tuple[int, ...],
        retry_attempts: cython.int = 3,
        retry_wait_time: cython.double = 1.0,
    ) -> None:
        """Retry a sync or async block when `errors.MySQLError.errno` matches.

        Sync usage
        ----------
        ```python
        for attempt in RetryOnErrno(
            (2003, 2013, 1205),
            retry_attempts=3,
            retry_wait_time=1.0,
        ):
            with attempt:
                # Some SQL execution
        ```

        Async usage
        -----------
        ```python
        async for attempt in RetryOnErrno(
            (2003, 2013, 1205),
            retry_attempts=3,
            retry_wait_time=1.0,
        ):
            async with attempt:
                # Some SQL execution
        ```

        Retry policy:
        - Only `errors.MySQLError` is handled.
        - A retry is triggered only when `exc.errno` is contained in `errno_codes`.
        - If errno does not match, the original exception is raised immediately.

        Retry counting semantics:
        - `retry_attempts` is the number of retries **after** the initial call.
        - Total max executions = `1 + retry_attempts` when `retry_attempts > 0`.
        - If `retry_attempts <= 0`, retries are treated as infinite.

        Wait time semantics:
        - `retry_wait_time` is clamped to `0.0` when negative.
        - If `retry_wait_time == 0.0`, retries occur immediately (no sleep).

        Logging:
        - Each retry emits a warning with exception text, retry count, matched errno,
        and wait duration.

        :param errno_codes `<'tuple[int]'>`: MySQL errno values that should trigger retry.
            Must not be empty.
        :param retry_attempts `<'int'>`: Number of retries after the first attempt. Defaults to `3`.
            If `retry_attempts<=0`, retry indefinitely until success or non-retryable error.
        :param retry_wait_time `<'float'>`: Seconds to wait between retries. Defaults to `1.0`.
            Negative values are clamped to `0.0`.
        :raises errors.RetryValueError: If `errno_codes` is empty.
        """
        if len(errno_codes) == 0:
            raise errors.RetryValueError("errno_codes cannot be empty.")
        self._errno_set: set = set(errno_codes)
        self._retry_attempts = max(retry_attempts, 0)  # clamp to 0
        self._retry_wait_time = max(retry_wait_time, 0.0)  # clamp to 0.0
        self._retries = 0
        self._should_retry = True

    # Sync ------------------------------------------------------------------
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Self:
        if not self._should_retry:
            raise StopIteration

        self._should_retry = False
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # If block succeeded or exception is not retryable, do not suppress (re-raise if exception).
        self._should_retry = self._should_retry_on_exc(exc_type, exc_val)
        if not self._should_retry:
            return False

        # Retryable exception occurred, log and sleep before next retry.
        self._retries += 1
        # fmt: off
        logger.warning(
            "%s, retry (%d) on errno [%s] in %.1f seconds",
            exc_val, self._retries, exc_val.errno, self._retry_wait_time,
        )
        # fmt: on
        if self._retry_wait_time > 0.0:
            time.sleep(self._retry_wait_time)
        # Suppress current exception so the outer loop can retry.
        return True

    # Async -----------------------------------------------------------------
    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Self:
        if not self._should_retry:
            raise StopAsyncIteration

        self._should_retry = False
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # If block succeeded or exception is not retryable, do not suppress (re-raise if exception).
        self._should_retry = self._should_retry_on_exc(exc_type, exc_val)
        if not self._should_retry:
            return False

        # Retryable exception occurred, log and sleep before next retry.
        self._retries += 1
        # fmt: off
        logger.warning(
            "%s, retry (%d) on errno [%s] in %.1f seconds",
            exc_val, self._retries, exc_val.errno, self._retry_wait_time,
        )
        # fmt: on
        if self._retry_wait_time > 0.0:
            await asyncio.sleep(self._retry_wait_time)
        # Suppress current exception so the outer loop can retry.
        return True

    # Internal --------------------------------------------------------------
    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def _should_retry_on_exc(self, exc_type: object, exc_val: object) -> cython.bint:
        # Block succeeded
        if exc_type is None or exc_val is None:
            return False

        # Retry limit reached, re-raise
        if self._retry_attempts > 0 and self._retries >= self._retry_attempts:
            return False

        # Not MySQLError, re-raise
        if not isinstance(exc_val, errors.MySQLError):
            return False

        # MySQL error, but not in matching the retry error number, re-raise
        errno = exc_val.errno
        if not set_contains(self._errno_set, errno):
            return False

        # Should retry
        return True


# Retry on error -----------------------------------------------------------------------------------------------
def retry_on_error(
    error_classes: tuple[type[Exception], ...],
    retry_attempts: cython.int = 3,
    retry_wait_time: cython.double = 1.0,
) -> Callable[[Callable[P, R] | Callable[P, Awaitable[R]]], Callable[P, Any]]:
    """Decorator that retries sync/async callables when raised exception type exactly matches `error_classes`.

    This decorator supports both synchronous and asynchronous functions:
    - For sync callables, retry wait uses `time.sleep(...)`.
    - For async callables, retry wait uses `await asyncio.sleep(...)`.

    Retry matching is **exact type only**:
    - A retry is triggered only when `type(exc)` is in `error_classes`.
    - Subclasses are **not** matched automatically.

    Retry counting semantics:
    - `retry_attempts` is the number of retries **after** the initial call.
    - Total max executions = `1 + retry_attempts` when `retry_attempts > 0`.
    - If `retry_attempts <= 0`, retries are treated as infinite.

    Wait time semantics:
    - `retry_wait_time` is clamped to `0.0` when negative.
    - If `retry_wait_time == 0.0`, retries happen immediately without sleeping.

    Logging:
    - Each retry emits a warning log including the exception, current retry count,
      and wait seconds.

    :param error_classes `<'tuple[type[Exception]]'>`: Exception classes that should trigger retry.
        Must not be empty. Matching is exact (`type(exc)`), not `isinstance(...)`.
    :param retry_attempts `<'int'>`: Number of retry attempts after the first call. Defaults to `3`.
        If `retry_attempts<=0`, retry indefinitely until success or non-matching exception.
    :param retry_wait_time `<'float'>`: Seconds to wait between retries. Defaults to `1.0`.
        Negative values are clamped to `0.0`.
    :return `<'Callable'>`: A decorator that wraps a sync/async callable with retry logic.
    :raises errors.RetryValueError: If `error_classes` is empty.

    Example
    -------
    >>> @retry_on_error((errors.OperationalError,), retry_attempts=5, retry_wait_time=0.5)
    ... def run_query() -> None:
    ...     ...
    """
    if len(error_classes) == 0:
        raise errors.RetryValueError("error_classes cannot be empty.")
    error_set: set = set(error_classes)
    retry_attempts = max(retry_attempts, 0)  # clamp to 0
    retry_wait_time = max(retry_wait_time, 0.0)  # clamp to 0.0

    def decorator(func: Callable[P, R] | Callable[P, Awaitable[R]]) -> Callable[P, Any]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> R:
            retries: cython.longlong = 0
            while True:
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc]
                except Exception as exc:
                    if retry_attempts > 0 and retries >= retry_attempts:
                        raise
                    exc_type = type(exc)
                    if not set_contains(error_set, exc_type):
                        raise
                    retries += 1
                    # fmt: off
                    logger.warning(
                        "%s, retry (%d) on error %s in %.1f seconds",
                        exc, retries, exc_type, retry_wait_time,
                    )
                    # fmt: on
                    if retry_wait_time > 0:
                        await asyncio.sleep(retry_wait_time)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> R:
            retries: cython.longlong = 0
            while True:
                try:
                    return func(*args, **kwargs)  # type: ignore[misc]
                except Exception as exc:
                    if retry_attempts > 0 and retries >= retry_attempts:
                        raise
                    exc_type = type(exc)
                    if not set_contains(error_set, exc_type):
                        raise
                    retries += 1
                    # fmt: off
                    logger.warning(
                        "%s, retry (%d) on error %s in %.1f seconds",
                        exc, retries, exc_type, retry_wait_time,
                    )
                    # fmt: on
                    if retry_wait_time > 0:
                        time.sleep(retry_wait_time)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@cython.cclass
class RetryOnError:
    """Retry a sync or async block when exception type matches.

    Sync usage
    ----------
    ```python
    for attempt in RetryOnError(
        (errors.OperationalError,),
        retry_attempts=3,
        retry_wait_time=1.0,
    ):
        with attempt:
            # Some SQL execution
    ```

    Async usage
    -----------
    ```python
    async for attempt in RetryOnError(
        (errors.OperationalError,),
        retry_attempts=3,
        retry_wait_time=1.0,
    ):
        async with attempt:
            # Some SQL execution
    ```
    """

    _error_set: set[type[Exception]]
    _retry_attempts: cython.int
    _retry_wait_time: cython.double
    _retries: cython.longlong
    _should_retry: cython.bint

    def __init__(
        self,
        error_classes: tuple[type[Exception], ...],
        retry_attempts: cython.int = 3,
        retry_wait_time: cython.double = 1.0,
    ) -> None:
        """Retry a sync or async block when exception type matches.

        Sync usage
        ----------
        ```python
        for attempt in RetryOnError(
            (errors.OperationalError,),
            retry_attempts=3,
            retry_wait_time=1.0,
        ):
            with attempt:
                # Some SQL execution
        ```

        Async usage
        -----------
        ```python
        async for attempt in RetryOnError(
            (errors.OperationalError,),
            retry_attempts=3,
            retry_wait_time=1.0,
        ):
            async with attempt:
                # Some SQL execution
        ```

        Retry matching is **exact type only**:
        - A retry is triggered only when `type(exc)` is in `error_classes`.
        - Subclasses are **not** matched automatically.

        Retry counting semantics:
        - `retry_attempts` is the number of retries **after** the initial call.
        - Total max executions = `1 + retry_attempts` when `retry_attempts > 0`.
        - If `retry_attempts <= 0`, retries are treated as infinite.

        Wait time semantics:
        - `retry_wait_time` is clamped to `0.0` when negative.
        - If `retry_wait_time == 0.0`, retries happen immediately without sleeping.

        Logging:
        - Each retry emits a warning log including the exception, current retry count,
          and wait seconds.

        :param error_classes `<'tuple[type[Exception]]'>`: Exception classes that should trigger retry.
            Must not be empty. Matching is exact (`type(exc)`), not `isinstance(...)`.
        :param retry_attempts `<'int'>`: Number of retry attempts after the first call. Defaults to `3`.
            If `retry_attempts<=0`, retry indefinitely until success or non-matching exception.
        :param retry_wait_time `<'float'>`: Seconds to wait between retries. Defaults to `1.0`.
            Negative values are clamped to `0.0`.
        """
        if len(error_classes) == 0:
            raise errors.RetryValueError("error_classes cannot be empty.")
        self._error_set: set = set(error_classes)
        self._retry_attempts = max(retry_attempts, 0)  # clamp to 0
        self._retry_wait_time = max(retry_wait_time, 0.0)  # clamp to 0.0
        self._retries = 0
        self._should_retry = True

    # Sync ------------------------------------------------------------------
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Self:
        if not self._should_retry:
            raise StopIteration

        self._should_retry = False
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Block succeeded or exception type not in retry set, do not suppress (re-raise if exception).
        self._should_retry = self._should_retry_on_exc(exc_type)
        if not self._should_retry:
            return False

        # Retryable exception occurred, log and sleep before next retry.
        self._retries += 1
        # fmt: off
        logger.warning(
            "%s, retry (%d) on error %s in %.1f seconds",
            exc_val, self._retries, exc_type, self._retry_wait_time,
        )
        # fmt: on
        if self._retry_wait_time > 0.0:
            time.sleep(self._retry_wait_time)
        # Suppress current exception so the outer loop can retry.
        return True

    # Async -----------------------------------------------------------------
    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Self:
        if not self._should_retry:
            raise StopAsyncIteration

        self._should_retry = False
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Block succeeded or exception type not in retry set, do not suppress (re-raise if exception).
        self._should_retry = self._should_retry_on_exc(exc_type)
        if not self._should_retry:
            return False

        # Retryable exception occurred, log and sleep before next retry.
        self._retries += 1
        # fmt: off
        logger.warning(
            "%s, retry (%d) on error %s in %.1f seconds",
            exc_val, self._retries, exc_type, self._retry_wait_time,
        )
        # fmt: on
        if self._retry_wait_time > 0.0:
            await asyncio.sleep(self._retry_wait_time)
        # Suppress current exception so the outer loop can retry.
        return True

    # Internal --------------------------------------------------------------
    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def _should_retry_on_exc(self, exc_type: object) -> cython.bint:
        # Block succeeded
        if exc_type is None:
            return False

        # Retry limit reached, re-raise
        if self._retry_attempts > 0 and self._retries >= self._retry_attempts:
            return False

        # Not matching the retry error type, re-raise
        if not set_contains(self._error_set, exc_type):
            return False

        # Should retry
        return True
