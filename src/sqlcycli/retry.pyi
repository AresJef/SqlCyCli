# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from typing import Callable, Awaitable, Any, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")

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
