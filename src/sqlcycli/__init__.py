import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from sqlcycli import aio, constants, errors, sqlfunc, sqlintvl
from sqlcycli._ssl import SSL
from sqlcycli._auth import AuthPlugin
from sqlcycli.retry import retry_on_errno, retry_on_error
from sqlcycli._optionfile import OptionFile
from sqlcycli.charset import Charset, Charsets, all_charsets
from sqlcycli.protocol import MysqlPacket, FieldDescriptorPacket
from sqlcycli.transcode import escape, ObjStr, CustomEscapeType, BIT, JSON
from sqlcycli.connection import (
    Cursor,
    DictCursor,
    DfCursor,
    SSCursor,
    SSDictCursor,
    SSDfCursor,
    BaseConnection,
    Connection,
)
from sqlcycli.aio.pool import Pool, PoolConnection, PoolSyncConnection
from sqlcycli._connect import connect, create_pool


__all__ = [
    # Module
    "aio",
    "constants",
    "errors",
    "sqlfunc",
    "sqlintvl",
    # Class
    "AuthPlugin",
    "OptionFile",
    "SSL",
    "Charset",
    "Charsets",
    "MysqlPacket",
    "FieldDescriptorPacket",
    "Cursor",
    "DictCursor",
    "DfCursor",
    "SSCursor",
    "SSDictCursor",
    "SSDfCursor",
    "BaseConnection",
    "Connection",
    "Pool",
    "PoolConnection",
    "PoolSyncConnection",
    # Type
    "ObjStr",
    "CustomEscapeType",
    "BIT",
    "JSON",
    # Function
    "all_charsets",
    "escape",
    "connect",
    "create_pool",
    # Decorator
    "retry_on_errno",
    "retry_on_error",
]
