# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.sqlcycli import utils  # type: ignore

# Python imports
from sqlcycli import utils
from sqlcycli.constants import ER


# MySQL Exceptions -------------------------------------------------------------------------------
class MySQLError(Exception):
    """Base class for all exceptions raised by SQLCycli."""

    def __init__(self, *args, errno: int | None = None) -> None:
        if errno is None:
            super().__init__(*args)
        else:
            super().__init__(errno, *args)
        self._errno = errno

    @property
    def errno(self) -> int | None:
        """Get the error number `<'int/None'>`."""
        return self._errno


class Warning(Warning, MySQLError):
    """Exception raised for important warnings like data truncations
    while inserting, etc."""


class Error(MySQLError):
    """Exception that is the base class of all other error exceptions
    (not Warning)."""


class InterfaceError(Error):
    """Exception raised for errors that are related to the database
    interface rather than the database itself."""


class DatabaseError(Error):
    """Exception raised for errors that are related to the
    database."""


class DataError(DatabaseError):
    """Exception raised for errors that are due to problems with the
    processed data like division by zero, numeric value out of range,
    etc."""


class OperationalError(DatabaseError):
    """Exception raised for errors that are related to the database's
    operation and not necessarily under the control of the programmer,
    e.g. an unexpected disconnect occurs, the data source name is not
    found, a transaction could not be processed, a memory allocation
    error occurred during processing, etc."""


class OperationalTableNotExistsError(OperationalError):
    """Exception raised when a table does not exist."""


class OperationalTimeoutError(OperationalError, TimeoutError):
    """Exception raised when an operation times out."""


class OperationalUnknownCommandError(OperationalError):
    """Exception raised when an unknown command is received."""


class IntegrityError(DatabaseError):
    """Exception raised when the relational integrity of the database
    is affected, e.g. a foreign key check fails, duplicate key,
    etc."""


class InternalError(DatabaseError):
    """Exception raised when the database encounters an internal
    error, e.g. the cursor is not valid anymore, the transaction is
    out of sync, etc."""


class ProgrammingError(DatabaseError):
    """Exception raised for programming errors, e.g. table not found
    or already exists, syntax error in the SQL statement, wrong number
    of parameters specified, etc."""


class NotSupportedError(DatabaseError):
    """Exception raised in case a method or database API was used
    which is not supported by the database, e.g. requesting a
    .rollback() on a connection that does not support transaction or
    has transactions turned off."""


MYSQL_ERROR_MAP: dict[int, Exception] = {}


def _map_error(exc, *error_codes):
    for error in error_codes:
        MYSQL_ERROR_MAP[error] = exc


_map_error(
    ProgrammingError,
    ER.SYNTAX_ERROR,
    ER.PARSE_ERROR,
    ER.NO_SUCH_TABLE,
    ER.WRONG_DB_NAME,
    ER.WRONG_TABLE_NAME,
    ER.FIELD_SPECIFIED_TWICE,
    ER.INVALID_GROUP_FUNC_USE,
    ER.UNSUPPORTED_EXTENSION,
    ER.TABLE_MUST_HAVE_COLUMNS,
    ER.CANT_DO_THIS_DURING_AN_TRANSACTION,
    ER.WRONG_DB_NAME,
    ER.WRONG_COLUMN_NAME,
    ER.WRONG_AUTO_KEY,
)
_map_error(
    DataError,
    ER.WARN_DATA_TRUNCATED,
    ER.WARN_NULL_TO_NOTNULL,
    ER.WARN_DATA_OUT_OF_RANGE,
    ER.NO_DEFAULT,
    ER.PRIMARY_CANT_HAVE_NULL,
    ER.DATA_TOO_LONG,
    ER.DATETIME_FUNCTION_OVERFLOW,
    ER.TRUNCATED_WRONG_VALUE_FOR_FIELD,
    ER.ILLEGAL_VALUE_FOR_TYPE,
)
_map_error(
    IntegrityError,
    ER.DUP_ENTRY,
    ER.NO_REFERENCED_ROW,
    ER.NO_REFERENCED_ROW_2,
    ER.ROW_IS_REFERENCED,
    ER.ROW_IS_REFERENCED_2,
    ER.CANNOT_ADD_FOREIGN,
    ER.BAD_NULL_ERROR,
)
_map_error(
    NotSupportedError,
    ER.WARNING_NOT_COMPLETE_ROLLBACK,
    ER.NOT_SUPPORTED_YET,
    ER.FEATURE_DISABLED,
    ER.UNKNOWN_STORAGE_ENGINE,
)
_map_error(
    OperationalError,
    ER.DB_CREATE_EXISTS,
    ER.DB_DROP_EXISTS,
    ER.TABLE_EXISTS_ERROR,
    ER.BAD_TABLE_ERROR,
    ER.DUP_FIELDNAME,
    ER.CANT_DROP_FIELD_OR_KEY,
    ER.ACCESS_DENIED_ERROR,
    ER.DBACCESS_DENIED_ERROR,
    ER.TABLEACCESS_DENIED_ERROR,
    ER.COLUMNACCESS_DENIED_ERROR,
    ER.CONSTRAINT_FAILED,
    ER.CON_COUNT_ERROR,
    ER.LOCK_DEADLOCK,
)
_map_error(OperationalTableNotExistsError, ER.NO_SUCH_TABLE)
_map_error(
    OperationalTimeoutError,
    ER.LOCK_WAIT_TIMEOUT,
    ER.STATEMENT_TIMEOUT,
    ER.QUERY_TIMEOUT,
)
_map_error(OperationalUnknownCommandError, ER.UNKNOWN_COM_ERROR)

del _map_error, ER


@cython.ccall
@cython.exceptval(-1, check=False)
def raise_mysql_exception(
    data: cython.p_const_char,
    data_size: cython.Py_ssize_t,
) -> cython.bint:
    """Raise the MySQL exception based on the given data.

    :param data `<'char*/bytes'>`: The MySQL data contains the exception information.
    :param size `<'int'>`: The length of the data.
    """
    # https://dev.mysql.com/doc/dev/mysql-server/latest/page_protocol_basic_err_packet.html
    # Error packet has optional sqlstate that is 5 bytes and starts with '#'.
    if data_size < 5:
        error: bytes = data
        raise InternalError(error, errno=0)

    errno: cython.int = utils.unpack_int16(data, 1)
    if data[3] == 0x23:  # '#'
        error: bytes = data[9:data_size]
    else:
        error: bytes = data[3:data_size]
    error_msg = error.decode("utf8", "replace")
    error_cls = MYSQL_ERROR_MAP.get(errno, None)
    if error_cls is None:
        error_cls = InternalError if errno < 1000 else OperationalError
    raise error_cls(error_msg, errno=errno)


# Base Exceptions ---------------------------------------------------------------------------------
class MySQLTypeError(MySQLError, TypeError):
    """Raised when a type is invalid."""


class MySQLIndexError(MySQLError, IndexError):
    """Raised when an index is invalid."""


class MySQLValueError(MySQLError, ValueError):
    """Raised when a value is invalid."""


# Charset Exceptions ------------------------------------------------------------------------------
class CharsetError(MySQLError):
    """Base class for all exceptions raised by Charset."""


class CharsetNotFoundError(CharsetError, ProgrammingError):
    """Raised when a charset is not found."""


# Transcode Exceptions ----------------------------------------------------------------------------
class TranscodeError(MySQLTypeError, MySQLValueError):
    """Base class for all exceptions raised by Transcode."""


class EscapeError(TranscodeError, NotSupportedError):
    """Raised when an escape type is not supported."""


class EscapeTypeError(EscapeError, MySQLTypeError):
    """Raised when an escape type is invalid."""


class EscapeValueError(EscapeError, MySQLValueError):
    """Raised when an escape value is invalid."""


class DecodeError(TranscodeError, NotSupportedError):
    """Raised when a decode type is not supported."""


class DecodeTypeError(DecodeError, MySQLTypeError):
    """Raised when a decode type is invalid."""


class DecodeValueError(DecodeError, MySQLValueError):
    """Raised when a decode value is invalid."""


class SQLFunctionError(TranscodeError, ProgrammingError):
    """Raised when a SQL function is not supported."""


# Protocol Exceptions ----------------------------------------------------------------------------
class ProtocolError(MySQLError):
    """Base class for all exceptions raised by Protocol."""


class MysqlPacketError(ProtocolError):
    """Raised when a MySQL packet is invalid."""


class MysqlPacketCursorError(MysqlPacketError, MySQLIndexError):
    """Raised when a cursor is invalid."""


class AuthenticationError(MysqlPacketError, OperationalError):
    """Raised when an authentication packet is invalid."""


# Connection Exceptions --------------------------------------------------------------------------
class ConnectionError(MySQLError):
    """Base class for all exceptions raised by Connection."""


class ConnectionValueError(ConnectionError, MySQLValueError):
    """Raised when a value is invalid."""


class ConnectionFileNotFoundError(ConnectionError, FileNotFoundError):
    """Raised when a file is not found."""


class InvalidConnetionArgumentError(ConnectionValueError, ProgrammingError):
    """Raised when a connection value is invalid."""


class InvalidOptionFileError(InvalidConnetionArgumentError):
    """Raised when a MySQL option file is invalid."""


class OptionFileNotFoundError(InvalidOptionFileError, ConnectionFileNotFoundError):
    """Raised when a MySQL option file is not found."""


class InvalidSSLConfigError(InvalidConnetionArgumentError):
    """Raised when an SSL configuration is invalid."""


class SSLConfigFileNotFoundError(InvalidSSLConfigError, ConnectionFileNotFoundError):
    """Raised when an SSL configuration file is not found."""


class InvalidAuthPluginError(InvalidConnetionArgumentError):
    """Raised when an authentication plugin is invalid."""


class LocalFileNotFoundError(ConnectionFileNotFoundError, OperationalError):
    """Raised when local file not found."""


class CommandOutOfSyncError(ConnectionError, OperationalError):
    """Raised when a command is out of sync."""


class OpenConnectionError(ConnectionError, OperationalError):
    """Raised when a connection is not established."""


class ConnectionLostError(ConnectionError, OperationalError):
    """Raised when a connection is disconnected."""


class ConnectionClosedError(ConnectionError, InterfaceError):
    """Raised when a connection is closed."""


# Cursor Exceptions ------------------------------------------------------------------------------
class CursorError(MySQLError):
    """Base class for all exceptions raised by Cursor."""


class CursorIndexError(CursorError, MySQLIndexError):
    """Raised when an index is invalid."""


class CursorValueError(CursorError, MySQLValueError):
    """Raised when a value is invalid."""


class InvalidCursorIndexError(CursorIndexError, ProgrammingError):
    """Raised when an index is invalid."""


class InvalidCursorArgumentError(CursorValueError, ProgrammingError):
    """Raised when a cursor value is invalid."""


class InvalidSQLArgumentErorr(InvalidCursorArgumentError):
    """Raised when a SQL argument is invalid."""


class CursorNotExecutedError(CursorError, ProgrammingError):
    """Raised when a cursor is not executed."""


class CursorClosedError(CursorError, ProgrammingError):
    """Raised when a cursor is closed."""


# Connection Exceptions --------------------------------------------------------------------------
class PoolError(ConnectionError):
    """Base class for all exceptions raised by Pool."""


class PoolValueError(PoolError, ConnectionValueError):
    """Raised when a value is invalid."""


class InvalidPoolArgumentError(PoolValueError, InvalidConnetionArgumentError):
    """Raised when a pool value is invalid."""


class PoolReleaseError(PoolError, ProgrammingError):
    """Raised when a connection is not owned by the pool."""


class PoolClosedError(PoolError, ProgrammingError):
    """Raised when a pool is closed."""


class PoolNotClosedError(PoolError, ProgrammingError, RuntimeError):
    """Raised when a pool is not closed properly."""
