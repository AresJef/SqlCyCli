# cython: language_level=3
from sqlcycli cimport connection as sync_conn
from sqlcycli.aio cimport connection as async_conn, pool as aio_pool

# Connection
cdef class ConnectionManager:
    cdef:
        # . connection
        sync_conn.BaseConnection _conn_sync
        async_conn.BaseConnection _conn_async
        # . arguments
        dict _kwargs
        object _cursor
        object _loop

# Pool
cdef class PoolManager:
    cdef:
        # . pool
        aio_pool.Pool _pool
        # . arguments
        dict _kwargs
        object _cursor

