# cython: language_level=3

# Retry on errno
cdef class RetryOnErrno:
    cdef:
        set _errno_set
        int _retry_attempts
        double _retry_wait_time
        long long _retries
        bint _should_retry
    cdef bint _should_retry_on_exc(self, object exc_type, object exc_val) except -1

# Retry on error
cdef class RetryOnError:
    cdef:
        set _error_set
        int _retry_attempts
        double _retry_wait_time
        long long _retries
        bint _should_retry
    cdef bint _should_retry_on_exc(self, object exc_type) except -1
