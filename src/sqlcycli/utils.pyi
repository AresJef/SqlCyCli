import cython
from re import Pattern
from sqlcycli.charset import Charset

# Constants
DEFAULT_USER: str
DEFUALT_CHARSET: str
MAX_CONNECT_TIMEOUT: int
DEFALUT_MAX_ALLOWED_PACKET: int
MAXIMUM_MAX_ALLOWED_PACKET: int
MAX_PACKET_LENGTH: int
MAX_STATEMENT_LENGTH: int
SERVER_VERSION_RE: Pattern
RE_INSERT_VALUES: Pattern
"""regex pattern: python-constants"""
INSERT_VALUES_RE: Pattern
"""regex pattern: cython-constants"""
NULL_COLUMN: cython.uchar
UNSIGNED_CHAR_COLUMN: cython.uchar
UNSIGNED_SHORT_COLUMN: cython.uchar
UNSIGNED_INT24_COLUMN: cython.uchar
UNSIGNED_INT64_COLUMN: cython.uchar

# Utils: string
def encode_str(data: str, encoding: cython.p_char | bytes) -> bytes:
    """(cfunc) Encode string to bytes using the 'encoding' with 'surrogateescape' error handling `<'bytes'>`.

    :param data `<'str'>`: String to encode.
    :param encoding `<'char*/bytes'>`: The encoding to use.
    :returns `<'bytes'>`: Encoded bytes.
    """

def decode_bytes(data: bytes, encoding: cython.p_char | bytes) -> str:
    """(cfunc) Decode bytes to string using specified encoding with 'surrogateescape' error handling `<'str'>`.

    :param data `<'bytes'>`: Bytes to decode.
    :param encoding `<'char*/bytes'>`: Encoding to use for decoding.
    :returns `<'str'>`: Decoded string.
    """

def decode_bytes_utf8(data: bytes) -> str:
    """(cfunc) Decode bytes to string using 'utf-8' encoding with 'surrogateescape' error handling `<'str'>`.

    :param data `<'bytes'>`: Bytes to decode with `'utf-8'` encoding.
    :returns `<'str'>`: Decoded string.
    """

def decode_bytes_ascii(data: bytes) -> str:
    """(cfunc) Decode bytes to string using 'ascii' encoding with 'surrogateescape' error handling `<'str'>`.

    :param data `<'bytes'>`: Bytes to decode with `'ascii'` encoding.
    :returns `<'str'>`: Decoded string.
    """

def decode_bytes_latin1(data: bytes) -> str:
    """(cfunc) Decode bytes to string using 'latin1' encoding with 'surrogateescape' error handling `<'str'>`.

    :param data `<'bytes'>`: Bytes to decode with `'latin1'` encoding.
    :returns `<'str'>`: Decoded string.
    """

def find_null_byte(data: cython.p_char | bytes, length: int, pos: int) -> int:
    """(cfunc) Find the next NULL ('\\0') byte in a buffer starting at `pos` `<'int'>`.

    The search is restricted to the range `[pos, length)`.
    If no NULL byte is found in this range, `-1` is returned.

    :param data `<'char*/bytes'>`: Pointer to the beginning of the buffer.
    :param length `<'int'>`: Total length of the buffer in bytes.
    :param pos `<'int'>`: Zero-based offset where the search starts.
    :return `<'int'>`: The index (relative to `data`) of the next NULL byte, or `-1` if no NULL byte is found.
    :raises `<'ValueError'>`: If `pos` is negative or `length` is negative.

    ## Precondition
    - `data[0..length-1]` must be a valid readable memory region.
    """

# Utils: Pack unsigned integer to bytes
def pack_I24B(i: int, b: int) -> bytes:
    """(cfunc) Pack a 24-bit little-endian unsigned integer and
    a trailing byte into a 4-byte sequence `<'bytes'>`.

    :param i `<'int'>`: The unsigned integer; only least significant
        24 bits are preserved (`0 <= i <= 16777215`).
    :param b `<'int'>`: The trailing byte (0 <= b <= 255).
    :returns `<'bytes'>`: The packed byte sequence.

    ## Equivalent
    >>> struct.pack("<I", i)[0:3] + struct.pack("<B", b)
    """

def pack_IB(i: int, b: int) -> bytes:
    """(cfunc) Pack a 32-bit little-endian unsigned integer and a trailing byte into a 5-byte sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack as a 32-bit little-endian unsigned value.
    :param b `<'int'>`: The trailing byte (0 <= b <= 255).
    :returns `<'bytes'>`: The packed 5-byte sequence.

    ## Equivalent
    >>> struct.pack("<IB", i, b)
    """

def pack_IIB23s(i1: int, i2: int, b: int) -> bytes:
    """(cfunc) Pack two 32-bit little-endian unsigned integers, a byte, and
    a 23-byte zero-filled string into a 32-byte sequence `<'bytes'>`.

    :param i1 `<'int'>`: The first 32-bit unsigned integer.
    :param i2 `<'int'>`: The second 32-bit unsigned integer.
    :param b `<'int'>`: The trailing byte (0 <= b <= 255).
    :returns `<'bytes'>`: The packed 32-byte sequence.

    ## Equivalent
    >>> struct.pack("<IIB23s", i1, i2, b, b"")
    """

def pack_uint8(i: int) -> bytes:
    """(cfunc) Pack an unsigned 8-bit integer into 1-byte sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        8 bits are preserved (`0 <= i <= 255`).
    :returns `<'bytes'>`: A 1-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<B", i)
    """

def pack_uint16(i: int) -> bytes:
    """(cfunc) Pack an unsigned 16-bit integer into 2-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        16 bits are preserved (`0 <= i <= 65535`).
    :returns `<'bytes'>`: A 2-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<H", i)
    """

def pack_uint24(i: int) -> bytes:
    """(cfunc) Pack an unsigned 24-bit integer into 3-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        24 bits are preserved (`0 <= i <= 16777215`).
    :returns `<'bytes'>`: A 3-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<I", i)[:3]
    """

def pack_uint32(i: int) -> bytes:
    """(cfunc) Pack an unsigned 32-bit integer into 4-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        32 bits are preserved (`0 <= i <= 4294967295`).
    :returns `<'bytes'>`: A 4-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<I", i)
    """

def pack_uint64(i: int) -> bytes:
    """(cfunc) Pack an unsigned 64-bit integer into 8-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        64 bits are preserved (`0 <= i <= 18446744073709551615`).
    :returns `<'bytes'>`: An 8-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<Q", i)
    """

def gen_length_encoded_integer(i: int) -> bytes:
    """(cfunc) Generate a MySQL Protocol `LengthEncodedInteger` `<'bytes'>`.

    :param i `<'int'>`: The integer to encode.
    :returns `<'bytes'>`: The length-encoded integer as bytes.

    ## Behavior
    - Values 0..250 are encoded as a single byte.
    - Values 251..65535 are encoded as 0xFC + 2-byte little-endian.
    - Values 65536..16777215 are encoded as 0xFD + 3-byte little-endian.
    - Larger values are encoded as 0xFE + 8-byte little-endian.

    ## Reference
    See: [LengthEncodedInteger](https://dev.mysql.com/doc/internals/en/integer.html#packet-Protocol::LengthEncodedInteger)
    """

# Utils: Pack integer to bytes
def pack_int8(i: int) -> bytes:
    """(cfunc) Pack a signed 8-bit integer into 1-byte sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        8 bits are preserved (`-128 <= i <= 127`).
    :returns `<'bytes'>`: A 1-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<b", i)
    """

def pack_int16(i: int) -> bytes:
    """(cfunc) Pack a signed 16-bit integer into 2-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        16 bits are preserved (`-32768 <= i <= 32767`).
    :returns `<'bytes'>`: A 2-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<h", i)
    """

def pack_int24(i: int) -> bytes:
    """(cfunc) Pack a signed 24-bit integer into 3-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        24 bits are preserved (`-8388608 <= i <= 8388607`).
    :returns `<'bytes'>`: A 3-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<i", i)[:3]
    """

def pack_int32(i: int) -> bytes:
    """(cfunc) Pack a signed 32-bit integer into 4-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        32 bits are preserved (`-2147483648 <= i <= 2147483647`).
    :returns `<'bytes'>`: A 4-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<i", i)
    """

def pack_int64(i: int) -> bytes:
    """(cfunc) Pack a signed 64-bit integer into 8-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant
        64 bits are preserved (`-9223372036854775808 <= i <= 9223372036854775807`).
    :returns `<'bytes'>`: An 8-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<q", i)
    """

# Utils: Unpack unsigned integer
def unpack_uint8(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack an unsigned 8-bit integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the byte is read from.
    :returns `<'int'>`: The unpacked unsigned 8-bit integer.
    """

def unpack_uint16(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack an unsigned 16-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 1]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 16-bit integer starts.
    :returns `<'int'>`: The unpacked unsigned 16-bit integer.
    """

def unpack_uint24(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack an unsigned 24-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 2]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 24-bit integer starts.
    :returns `<'int'>`: The unpacked unsigned 24-bit integer.
    """

def unpack_uint32(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack an unsigned 32-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 3]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 32-bit integer starts.
    :returns `<'int'>`: The unpacked unsigned 32-bit integer.
    """

def unpack_uint64(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack an unsigned 64-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 7]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 64-bit integer starts.
    :returns `<'int'>`: The unpacked unsigned 64-bit integer.
    """

# Utils: Unpack signed integer
def unpack_int8(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack a signed 8-bit integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the byte is read from.
    :returns `<'int'>`: The unpacked signed 8-bit integer.
    """

def unpack_int16(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack a signed 16-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 1]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'const char*'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 16-bit integer starts.
    :returns `<'int'>`: The unpacked signed 16-bit integer.
    """

def unpack_int24(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack a signed 24-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 2]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'const char*'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 24-bit integer starts.
    :returns `<'int'>`: The unpacked signed 24-bit integer.
    """

def unpack_int32(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack a signed 32-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 3]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'const char*'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 32-bit integer starts.
    :returns `<'int'>`: The unpacked signed 32-bit integer.
    """

def unpack_int64(data: cython.p_const_char | bytes, pos: int) -> int:
    """(cfunc) Unpack a signed 64-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 7]` must be within a valid readable buffer;
        otherwise leading to undefined behavior.

    :param data `<'const char*'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 64-bit integer starts.
    :returns `<'int'>`: The unpacked signed 64-bit integer.
    """

# Utils: Query
def format_sql(sql: str, args: str | tuple) -> str:
    """Format the sql with the arguments `<'str'>`.

    :param sql `<'str'>`: The sql to format.
    :param args `<'str/tuple'>`: The arguments to bound to the SQL.
    :raises `<'InvalidSQLArgsErorr'>`: If any error occurs.
    """

# Utils: Connection
def gen_connect_attrs(attrs: list[str]) -> bytes:
    """Generate connection attributes bytes from the given list of attribute strings `<'bytes'>`.

    :param attrs `<'list[str]'>`: A list of strings, each containing the connection attribute information.
    :returns `<'bytes'>`: The generated connection attributes as bytes.
    """

DEFAULT_CONNECT_ATTRS: bytes

# Utils: Argument Validator
def validate_arg_str(arg: str | None, arg_name: str, default: str | None) -> str | None:
    """Normalize an argument to a non-empty string, using a default when appropriate `<'str'>`.

    :param arg `<'str/None'>`: The argument value to validate.
    :param arg_name `<'str'>`: The argument name (for error messages).
    :param default `<'str/None'>`: The fallback value returned when `arg` is None or an empty string.
    :returns `<'str'>`: The validated string or the default value.
    :raises `<'InvalidConnectionArgsError'>`: If `arg` is neither None nor a string.

    ## Behavior
    - If `arg` is None, return `default`.
    - If `arg` is a string:
        * return it if non-empty,
        * otherwise return `default`.
    - Any other type raises `InvalidConnectionArgsError`.
    """

def validate_arg_int(
    arg: int | None,
    arg_name: str,
    min_value: int,
    max_value: int,
) -> int | None:
    """Validate that an argument is an integer within a given range or None `<'int/None'>`.

    :param arg `<'int/None'>`: The argument value to validate.
    :param arg_name `<'str'>`: The name of the argument (used in error messages).
    :param min_value `<'int'>`: The minimum allowed value (inclusive).
    :param max_value `<'int'>`: The maximum allowed value (inclusive).
    :returns `<'int/None'>`: The valid integer within the range, or None.
    :raises `<'InvalidConnectionArgsError'>`: If `arg` is neither None nor an int,
        or if it falls outside the `[min_value, max_value]` range.

    ## Behavior
    - If `arg` is None, return None.
    - If `arg` is an int and `min_value <= arg <= max_value`, return `arg` as-is.
    - Otherwise, raise `InvalidConnectionArgsError`.
    """

def validate_arg_bytes(
    arg: bytes | str | None,
    arg_name: str,
    encoding: cython.p_char | bytes,
    default: str | None,
) -> bytes | None:
    """Normalize an argument to a non-empty bytes value, with an optional default `<'bytes/None'>`.

    :param arg `<'str/bytes/None'>`: The argument value to validate.
    :param arg_name `<'str'>`: The argument name (used in error messages).
    :param encoding `<'char*/bytes'>`: The character encoding used when converting strings to bytes.
    :param default `<'str/None'>`: Optional default value used when `arg` is None or empty.
    :returns `<'bytes/None'>`: A non-empty bytes object, or None if no value or default is provided.
    :raises `<'InvalidConnectionArgsError'>`: If `arg` is not None/str/bytes,
        or if its type is str/bytes but invalid under the above rules.

    ## Behavior
    - If `arg` is None:
        * If `default` is not None, encode `default` using `encoding` and return bytes.
        * Otherwise, return None.
    - If `arg` is a str:
        * If non-empty, encode it using `encoding` and return bytes.
        * If empty, behave as if `arg` were None (use `default` or return None).
    - If `arg` is bytes:
        * If non-empty, return `arg` as-is (no copy).
        * If empty, behave as if `arg` were None (use `default` or return None).
    - For any other type, raise `InvalidConnectionArgsError`.
    """

def validate_charset(
    charset: str | None,
    collation: str | None,
    default_charset: str | None,
) -> Charset:
    """Resolve charset and optional collation into `<'Charset'>`.

    :param charset `<'str/None'>: Charset name.
    :param collation `<'str/None'>`: Collation name
    :param default_charset `<'str/None'>`: Fallback charset name used when `charset` is None/empty.
    :returns `<'Charset'>`: The resolved charset object.
    :raises `<'InvalidConnectionArgsError'>`: If `charset` or `collation` has
        an invalid type or is an empty string.
    :raises `<'CharsetNotFoundError'>` If the resolved charset or collation does not exist.

    ## Behavior
    - `charset`:
        * If None or an empty string, `default_charset` is used.
        * Otherwise, must be a non-empty string; invalid types are rejected.
    - `collation`:
        * If None or an empty string, use the default collation for `charset`.
        * If a non-empty string, resolve the specific charset & collation pair.
        * Invalid types are rejected.
    """

def validate_autocommit(autocommit: bool | None) -> int:
    """Normalize the `autocommit` option to a tri-state integer `<'int'>`.

    :param autocommit `<'bool/None'>`: The autocommit flag.
    :returns `<'int'>`: `-1`, `0`, or `1`, "default", "off", or "on" respectively.

    ## Behavior
    - If `autocommit` is None, return -1 (meaning: use server/default behavior).
    - Otherwise, convert `autocommit` to bool:
        * True returns 1 (autocommit ON)
        * False returns 0 (autocommit OFF)
    """

def validate_max_allowed_packet(
    max_allowed_packet: int | str | None,
    default: int,
    max_value: int,
) -> int:
    """Validate and normalize the `max_allowed_packet` option `<'int'>`.

    :param max_allowed_packet `<'str/int/None'>`: The max allowed packet size.
    :param default `<'int'>`: The default value used when `max_allowed_packet` is None.
    :param max_value `<'int'>`: The maximum allowed value for `max_allowed_packet`.
    :returns `<'int'>`: The validated max allowed packet size.

    ## Behavior
    - Accepted forms for `max_allowed_packet`:
        * `None`: return `default`.
        * `<'int'>`: interpreted as a raw byte value.
        * `<'str'>`: integer with optional unit suffix:
            - "n"                          → n bytes
            - "n[K|k]"  or "<n>[K|k][B|b]" → n * 1024 bytes
            - "n[M|m]"  or "<n>[M|m][B|b]" → n * 1024^2 bytes
            - "n[G|g]"  or "<n>[G|g][B|b]" → n * 1024^3 bytes
    - The final value must satisfy `1 <= value <= max_value`.
    """

def validate_sql_mode(sql_mode: str | None) -> str | None:
    """Validate and escape the `sql_mode` argument `<'str/None'>`.

    :param sql_mode `<'str/None'>`: The sql_mode value.
    :returns `<'str/None'>`: The escaped sql_mode string, or None.
    :raises `<'InvalidConnectionArgsError'>`: If `sql_mode` is neither None nor a string.
    """

def validate_ssl(ssl: object | None) -> object | None:
    """Validate and normalize the `ssl` argument `<'SSLContext/SSL/None'>`.

    :param ssl `<'SSLContext/SSL/None'>`: The ssl argument.
    :returns `<'SSLContext/None'>`: The SSLContext object or None.
    :raises `<'InvalidConnectionArgsError'>`: If `ssl` is neither None, SSL, nor SSLContext.
    """
