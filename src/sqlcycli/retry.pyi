# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from typing import Callable, Awaitable, Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def retry_on_errno(
    errno_codes: tuple[int, ...],
    retry_attempts: int = 3,
    retry_wait_time: float = 1.0,
) -> Callable[[Callable[P, R] | Callable[P, Awaitable[R]]], Callable[P, Any]]: ...
