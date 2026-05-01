# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from typing_extensions import Self
from typing import Callable, Awaitable, Any, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")

# Retry on errno -----------------------------------------------------------------------------------------------
@overload
def retry_on_errno(
    errno_codes: tuple[int, ...],
    retry_attempts: int = 3,
    retry_wait_time: float = 1.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def retry_on_errno(
    errno_codes: tuple[int, ...],
    retry_attempts: int = 3,
    retry_wait_time: float = 1.0,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...
def retry_on_errno(
    errno_codes: tuple[int, ...],
    retry_attempts: int = 3,
    retry_wait_time: float = 1.0,
) -> Callable[[Callable[P, R] | Callable[P, Awaitable[R]]], Callable[P, Any]]: ...

class RetryOnErrno:
    def __init__(
        self,
        errno_codes: tuple[int, ...],
        retry_attempts: int = 3,
        retry_wait_time: float = 1.0,
    ) -> None: ...
    # . sync
    def __iter__(self) -> Self: ...
    def __next__(self) -> Self: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...
    # . async
    def __aiter(self) -> Self: ...
    async def __anext__(self) -> Self: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool: ...

# Retry on error -----------------------------------------------------------------------------------------------
@overload
def retry_on_error(
    error_classes: tuple[type[Exception], ...],
    retry_attempts: int = 3,
    retry_wait_time: int = 1.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def retry_on_error(
    error_classes: tuple[type[Exception], ...],
    retry_attempts: int = 3,
    retry_wait_time: int = 1.0,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...
def retry_on_error(
    error_classes: tuple[type[Exception], ...],
    retry_attempts: int = 3,
    retry_wait_time: int = 1.0,
) -> Callable[[Callable[P, R] | Callable[P, Awaitable[R]]], Callable[P, Any]]: ...

class RetryOnError:
    def __init__(
        self,
        error_classes: tuple[type[Exception], ...],
        retry_attempts: int = 3,
        retry_wait_time: float = 1.0,
    ) -> None: ...
    # . sync
    def __iter__(self) -> Self: ...
    def __next__(self) -> Self: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...
    # . async
    def __aiter(self) -> Self: ...
    async def __anext__(self) -> Self: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool: ...
