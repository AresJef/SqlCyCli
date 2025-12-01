# cython: language_level=3

# Cython imports
from cpython.bytes cimport PyBytes_Size as bytes_len
from cpython.unicode cimport (
    PyUnicode_ReadChar as str_read,
    PyUnicode_GET_LENGTH as str_len,
    PyUnicode_Substring as str_substr,
)
from sqlcycli.transcode cimport escape
from sqlcycli._ssl cimport is_ssl, is_ssl_ctx
from sqlcycli.charset cimport Charset, _charsets

# Python imports
import re
from sqlcycli import errors

# Constants -----------------------------------------------------------------------------------
#: Regular expression for :meth:`Cursor.executemany`.
#: executemany only supports simple bulk insert.
#: You can use it to load large dataset.
RE_INSERT_VALUES: re.Pattern = re.compile(
    r"\s*((?:INSERT|REPLACE)\b.+\bVALUES?\s*)"  # prefix: INSERT INTO ... VALUES
    + r"(\(\s*(?:%s|%\(.+\)s)\s*(?:,\s*(?:%s|%\(.+\)s)\s*)*\))"  # placeholders: (%s, %s, ...)
    + r"(\s*(?:AS\b\s.+)?\s*(?:ON DUPLICATE\b.+)?);?\s*\Z",  # suffix: AS ... ON DUPLICATE ...
    re.IGNORECASE | re.DOTALL,
)

cdef str DEFAULT_USER
try:
    import getpass

    DEFAULT_USER: str = getpass.getuser()
    del getpass
except (ImportError, KeyError):
    #: KeyError occurs when there's no entry
    #: in OS database for a current user.
    DEFAULT_USER: str = None

cdef:
    str DEFUALT_CHARSET = "utf8mb4"
    int MAX_CONNECT_TIMEOUT = 31_536_000  # 1 year
    int DEFALUT_MAX_ALLOWED_PACKET = 16_777_216  # 16MB
    int MAXIMUM_MAX_ALLOWED_PACKET = 1_073_741_824  # 1GB
    unsigned int MAX_PACKET_LENGTH = 2**24 - 1
    #: Max statement size which :meth:`executemany` generates.
    #: Max size of allowed statement is max_allowed_packet - packet_header_size.
    #: Default value of max_allowed_packet is 1048576.
    unsigned int MAX_STATEMENT_LENGTH = 1024000
    #: Regular expression for :meth:`Cursor.executemany`.
    object INSERT_VALUES_RE = RE_INSERT_VALUES
    #: Regular expression for server version.
    object SERVER_VERSION_RE = re.compile(r".*?(\d+)\.(\d+)\.(\d+).*?")
    # The following values are for the first byte
    # value of MySQL length encoded integer.
    unsigned char NULL_COLUMN = 251
    unsigned char UNSIGNED_CHAR_COLUMN = 251
    unsigned char UNSIGNED_SHORT_COLUMN = 252
    unsigned char UNSIGNED_INT24_COLUMN = 253
    unsigned char UNSIGNED_INT64_COLUMN = 254

# Utils: Query
cpdef str format_sql(str sql, object args):
    """Format the sql with the arguments `<'str'>`.

    :param sql `<'str'>`: The sql to format.
    :param args `<'str/tuple'>`: The arguments to bound to the SQL.
    :raises `<'InvalidSQLArgsErorr'>`: If any error occurs.
    """
    try:
        return sql % args
    except Exception as err:
        raise errors.InvalidSQLArgumentErorr(
            "Failed to format SQL:\n%s\n"
            "With %s arguments:\n%r\n"
            "Error: %s" % (sql, type(args), args, err)
        ) from err

# Utils: Connection
cpdef bytes gen_connect_attrs(list attrs):
    """Generate connection attributes bytes from the given list of attribute strings `<'bytes'>`.

    :param attrs `<'list[str]'>`: A list of strings, each containing the connection attribute information.
    :returns `<'bytes'>`: The generated connection attributes as bytes.
    """
    cdef list arr = []
    cdef bytes attr_bytes
    for attr in attrs:
        if not isinstance(attr, str):
            raise errors.InvalidConnetionArgumentError(
                "Connection attribute must be <'str'>, "
                "instead got %r %s." % (attr, type(attr))
            )
        attr_bytes = encode_str(attr, b"utf8")
        arr.append(gen_length_encoded_integer(bytes_len(attr_bytes)))
        arr.append(attr_bytes)
    return b"".join(arr)

cdef bytes DEFAULT_CONNECT_ATTRS = gen_connect_attrs(
    ["_client_name", "sqlcycli", "_client_version", "0.0.0", "_pid"]
)

# Utils: Argument Validator
cpdef str validate_arg_str(object arg, str arg_name, str default):
    """Normalize an argument to a non-empty string, using a default when appropriate `<'str/None'>`.

    :param arg `<'str/None'>`: The argument value to validate.
    :param arg_name `<'str'>`: The argument name (for error messages).
    :param default `<'str/None'>`: The fallback value returned when `arg` is None or an empty string.
    :returns `<'str'>`: The validated string or the default value.
    :raises `<'InvalidConnetionArgumentError'>`: If `arg` is neither None nor a string.

    ## Behavior
    - If `arg` is None, return `default`.
    - If `arg` is a string:
        * return it if non-empty,
        * otherwise return `default`.
    - Any other type raises `InvalidConnetionArgumentError`.
    """
    if arg is None:
        return default
    if not isinstance(arg, str):
        raise errors.InvalidConnetionArgumentError(
            "Invalid '%s' argument %r %s.\n"
            "Expects a non-empty <'str'> or None." 
            % (arg_name, arg, type(arg))
        )
    return arg if str_len(arg) > 0 else default

cpdef object validate_arg_int(object arg, str arg_name, long long min_value, long long max_value):
    """Validate that an argument is an integer within a given range or None `<'int/None'>`.

    :param arg `<'int/None'>`: The argument value to validate.
    :param arg_name `<'str'>`: The name of the argument (used in error messages).
    :param min_value `<'int'>`: The minimum allowed value (inclusive).
    :param max_value `<'int'>`: The maximum allowed value (inclusive).
    :returns `<'int/None'>`: The valid integer within the range, or None.
    :raises `<'InvalidConnetionArgumentError'>`: If `arg` is neither None nor an int,
        or if it falls outside the `[min_value, max_value]` range.

    ## Behavior
    - If `arg` is None, return None.
    - If `arg` is an int and `min_value <= arg <= max_value`, return `arg` as-is.
    - Otherwise, raise `InvalidConnetionArgumentError`.
    """
    if arg is None:
        return None
    if not isinstance(arg, int):
        raise errors.InvalidConnetionArgumentError(
            "Invalid '%s' argument %r %s.\n"
            "Expects an <'int'> or None." 
            % (arg_name, arg, type(arg))
        )
    cdef long long value = int(arg)
    if not min_value <= value <= max_value:
        raise errors.InvalidConnetionArgumentError(
            "Invalid '%s' argument %d.\n"
            "Expects an integer between %d and %d." 
            % (arg_name, value, min_value, max_value)
        )
    return arg

cpdef bytes validate_arg_bytes(object arg, str arg_name, const char* encoding, str default):
    """Normalize an argument to a non-empty bytes value, with an optional default `<'bytes/None'>`.

    :param arg `<'str/bytes/None'>`: The argument value to validate.
    :param arg_name `<'str'>`: The argument name (used in error messages).
    :param encoding `<'char*/bytes'>`: The character encoding used when converting strings to bytes.
    :param default `<'str/None'>`: Optional default value used when `arg` is None or empty.
    :returns `<'bytes/None'>`: A non-empty bytes object, or None if no value or default is provided.
    :raises `<'InvalidConnetionArgumentError'>`: If `arg` is not None/str/bytes, 
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
    - For any other type, raise `InvalidConnetionArgumentError`.
    """
    if arg is None:
        if default is not None:
            return encode_str(default, encoding)
        return None
    if isinstance(arg, str):
        if str_len(arg) > 0:
            return encode_str(arg, encoding)
        if default is not None:
            return encode_str(default, encoding)
        return None
    if isinstance(arg, bytes):
        if bytes_len(arg) > 0:
            return arg
        if default is not None:
            return encode_str(default, encoding)
        return None
    raise errors.InvalidConnetionArgumentError(
        "Invalid '%s' argument %r %s.\n"
        "Expects a non-empty <'str'>, <'bytes'> or None." 
        % (arg_name, arg, type(arg))
    )

cpdef Charset validate_charset(object charset, object collation, str default_charset):
    """Resolve charset and optional collation into `<'Charset'>`.

    :param charset `<'str/None'>: Charset name.
    :param collation `<'str/None'>`: Collation name
    :param default_charset `<'str/None'>`: Fallback charset name used when `charset` is None/empty.
    :returns `<'Charset'>`: The resolved charset object.
    :raises `<'InvalidConnetionArgumentError'>`: If `charset` or `collation` has 
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
    cdef:
        str ch = validate_arg_str(charset, "charset", default_charset)
        str cl = validate_arg_str(collation, "collation", None)
    if cl is None:
        return _charsets.by_name(ch)
    else:
        return _charsets.by_name_n_collation(ch, cl)

cpdef int validate_autocommit(object autocommit) except -2:
    """Normalize the `autocommit` option to a tri-state integer `<'int'>`.

    :param autocommit `<'bool/None'>`: The autocommit flag.
    :returns `<'int'>`: `-1`, `0`, or `1`, "default", "off", or "on" respectively.

    ## Behavior
    - If `autocommit` is None, return -1 (meaning: use server/default behavior).
    - Otherwise, convert `autocommit` to bool:
        * True returns 1 (autocommit ON)
        * False returns 0 (autocommit OFF)
    """
    if autocommit is None:
        return -1
    else:
        return 1 if bool(autocommit) else 0

cpdef int validate_max_allowed_packet(object max_allowed_packet, int default, int max_value):
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
    # Argument is None
    if max_allowed_packet is None:
        return default

    # Argument is integer
    cdef: 
        long long  value, multiplier
        Py_ssize_t size
        Py_UCS4    ch
    if isinstance(max_allowed_packet, int):
        try:
            value = int(max_allowed_packet)
        except Exception as err:
            raise errors.InvalidConnetionArgumentError(
                "Invalid 'max_allowed_packet' argument %r %s.\n"
                "Expects <'str'>, <'int'> or None."
                % (max_allowed_packet, type(max_allowed_packet))
            ) from err

    # Argument is string
    elif isinstance(max_allowed_packet, str):
        size = str_len(max_allowed_packet)
        try:
            if size < 1:
                raise ValueError("not enough characters")
            # Parse unit suffix
            ch = str_read(max_allowed_packet, size - 1)
            # . skip [B] suffix
            if ch in ("B", "b"):
                size -= 1
                if size < 1:
                    raise ValueError("not enough characters.")
                ch = str_read(max_allowed_packet, size - 1)
            # . K (KiB) suffix
            if ch in ("K", "k"):
                size -= 1; multiplier = 1024
            # . M (MiB) suffix
            elif ch in ("M", "m"):
                size -= 1; multiplier = 1024 * 1024
            # . G (GiB) suffix
            elif ch in ("G", "g"):
                size -= 1; multiplier = 1024 * 1024 * 1024
            # . No suffix
            else:
                multiplier = 1
            # Parse integer
            if size < 1:
                raise ValueError("not enough characters.")
            value = int(str_substr(max_allowed_packet, 0, size))
        except Exception as err:
            raise errors.InvalidConnetionArgumentError(
                "Invalid 'max_allowed_packet' argument %r %s.\nError: %s"
                % (max_allowed_packet, type(max_allowed_packet), err)
            ) from err
        value = value * multiplier

    # Invalid type
    else:
        raise errors.InvalidConnetionArgumentError(
            "Invalid 'max_allowed_packet' argument %r %s.\n"
            "Expects <'str'>, <'int'> or None."
            % (max_allowed_packet, type(max_allowed_packet))
        )

    # Validate range
    if not 1 <= value <= max_value:
        raise errors.InvalidConnetionArgumentError(
            "Invalid 'max_allowed_packet' argument %d.\n"
            "Value must be between 1 and %d bytes."
            % (value, max_value)
        )
    return <int> value
    
cpdef str validate_sql_mode(object sql_mode):
    """Validate and escape the `sql_mode` argument `<'str/None'>`.

    :param sql_mode `<'str/None'>`: The sql_mode value.
    :returns `<'str/None'>`: The escaped sql_mode string, or None.
    :raises `<'InvalidConnetionArgumentError'>`: If `sql_mode` is neither None nor a string.
    """
    if sql_mode is None:
        return None
    if not isinstance(sql_mode, str):
        raise errors.InvalidConnetionArgumentError(
            "Invalid 'sql_mode' argument %r %s.\n"
            "Expects a <'str'> or None." 
            % (sql_mode, type(sql_mode))
        )
    return escape(sql_mode, False, False) if str_len(sql_mode) > 0 else None

cpdef object validate_ssl(object ssl):
    """Validate and normalize the `ssl` argument `<'SSLContext/SSL/None'>`.

    :param ssl `<'SSLContext/SSL/None'>`: The ssl argument.
    :returns `<'SSLContext/None'>`: The SSLContext object or None.
    :raises `<'InvalidConnetionArgumentError'>`: If `ssl` is neither None, SSL, nor SSLContext.
    """
    if ssl is None:
        return None
    if is_ssl(ssl):
        return ssl.context if ssl else None
    if is_ssl_ctx(ssl):
        return ssl
    raise errors.InvalidConnetionArgumentError(
        "Invalid 'ssl' argument %r %s.\n"
        "Expects a <'SSLContext'>, <'SSL'> or None." 
        % (ssl, type(ssl))
    )

# Test --------------------------------------------------------------------------------------
cpdef bint _test_find_null_byte() except -1:
    cdef bytes data = b"hello\x00world\x00"
    cdef char* p = bytes_to_chars(data)
    cdef Py_ssize_t loc, length = bytes_len(data)

    loc = find_null_byte(p, length, 0)
    assert loc == 5, f"{data}: first null term should be at 5 instead got {loc}"
    loc = find_null_byte(p, length, 6)
    assert loc == 11, f"{data}: second null term at 11 instead of {loc}"
    loc = find_null_byte(p, length, length-1)
    assert loc == 11, f"{data}: second null term at 11 instead of {loc}"
    try:
        find_null_byte(p, length, -1)
    except ValueError as err:
        pass
    else:
        raise AssertionError("ValueError is not raised for negative pos.")
    try:
        find_null_byte(p, -1, 1)
    except ValueError as err:
        pass
    else:
        raise AssertionError("ValueError is not raised for negative pos.")
    loc = find_null_byte(p, length, length)
    assert loc == -1, f"out of range position should return -1 instead got {loc}"
    loc = find_null_byte(p, length, length + 1)
    assert loc == -1, f"out of range position should return -1 instead got {loc}"

    data = b"hello"
    p = bytes_to_chars(data)
    length = bytes_len(data)
    loc = find_null_byte(p, length, 0)
    assert loc == -1, f"No null term should return -1 instead got {loc}"

    print("Pass: find_null_byte".ljust(80))

cpdef bint _test_pack_I24B() except -1:
    import struct

    cdef unsigned long long i
    cdef unsigned char      b
    for i in range(0, 9_000_000, 100_000):
        for b in range(255):
            x = pack_I24B(i, b)
            y = struct.pack("<I", i)[0:3] + struct.pack("<B", b)
            assert x == y, f"pack_I24B {x} not equal {y}"

    print("Pass: pack_I24B")

    del struct

cpdef bint _test_pack_IB() except -1:
    import struct

    cdef unsigned long long i
    cdef unsigned char      b
    for i in range(0, 9_000_000, 100_000):
        for b in range(255):
            x = pack_IB(i, b)
            y = struct.pack("<IB", i, b)
            assert x == y, f"pack_IB {x} not equal {y}"

    print("Pass: pack_IB")

    del struct

cpdef bint _test_pack_IIB23s() except -1:
    import struct

    cdef unsigned long long i
    cdef unsigned char      b
    for i in range(0, 9_000_000, 100_000):
        for b in range(255):
            x = pack_IIB23s(i, i, b)
            y = struct.pack("<IIB23s", i, i, b, b"")
            assert x == y, f"pack_IIB23s {x} not equal {y}"

    print("Pass: pack_IIB23s")

    del struct

cpdef bint _test_pack_unpack_i8() except -1:
    import struct

    cdef int i, i_o
    cdef unsigned int u, u_o

    for i in range(-128, 128):
        x = pack_int8(i)
        y = struct.pack("<b", i)
        assert x == y, f"pack_int8 {x} not equal {y}"
        i_o = unpack_int8(x, 0)
        assert i == i_o, f"unpack_int8 {i_o} not equal {i}"

    print("Pass: pack_int8 & unpack_int8")

    for u in range(256):
        x = pack_uint8(u)
        y = struct.pack("<B", u)
        assert x == y, f"pack_uint8 {x} not equal {y}"
        u_o = unpack_uint8(x, 0)
        assert u == u_o, f"unpack_uint8 {u_o} not equal {u}"

    print("Pass: pack_uint8 & unpack_uint8")

    del struct

cpdef bint _test_pack_unpack_i16() except -1:
    import struct

    cdef int i, i_o
    cdef unsigned int u, u_o

    for i in range(-32768, 32768):
        x = pack_int16(i)
        y = struct.pack("<h", i)
        assert x == y, f"pack_int16 {x} not equal {y}"
        i_o = unpack_int16(x, 0)
        assert i == i_o, f"unpack_int16 {i_o} not equal {i}"

    print("Pass: pack_int16 & unpack_int16")

    for u in range(65536):
        x = pack_uint16(u)
        y = struct.pack("<H", u)
        assert x == y, f"pack_uint16 {x} not equal {y}"
        u_o = unpack_uint16(x, 0)
        assert u == u_o, f"unpack_uint16 {u_o} not equal {u}"

    print("Pass: pack_uint16 & unpack_uint16")

    del struct

cpdef bint _test_pack_unpack_i24() except -1:
    import struct

    cdef long long i, i_o
    cdef unsigned long long u, u_o

    for i in (-8388608, -8388607, -1, 0, 1, 8388606, 8388607):
        x = pack_int24(i)
        y = struct.pack("<i", i)[:3]
        assert x == y, f"pack_int24 {x} not equal {y}"
        i_o = unpack_int24(x, 0)
        assert i == i_o, f"unpack_int24 {i_o} not equal {i}"

    print("Pass: pack_int24 & unpack_int24")

    for u in (0, 1, 16777213, 16777214, 16777215):
        x = pack_uint24(u)
        y = struct.pack("<I", u)[:3]
        assert x == y, f"pack_uint24 {x} not equal {y}"
        u_o = unpack_uint24(x, 0)
        assert u == u_o, f"unpack_uint24 {u_o} not equal {u}"

    print("Pass: pack_uint24 & unpack_uint24")

    del struct

cpdef bint _test_pack_unpack_i32() except -1:
    import struct

    cdef long long i, i_o
    cdef unsigned long long u, u_o

    for i in (-2147483648, -2147483647, -1, 0, 1, 2147483646, 2147483647):
        x = pack_int32(i)
        y = struct.pack("<i", i)
        assert x == y, f"pack_int32 {x} not equal {y}"
        i_o = unpack_int32(x, 0)
        assert i == i_o, f"unpack_int32 {i_o} not equal {i}"

    print("Pass: pack_int32 & unpack_int32")

    for u in (0, 1, 4294967293, 4294967294, 4294967295):
        x = pack_uint32(u)
        y = struct.pack("<I", u)
        assert x == y, f"pack_uint32 {x} not equal {y}"
        u_o = unpack_uint32(x, 0)
        assert u == u_o, f"unpack_uint32 {u_o} not equal {u}"

    print("Pass: pack_uint32 & unpack_uint32")

    del struct
    
cpdef bint _test_pack_unpack_i64() except -1:
    import struct

    cdef long long i, i_o
    cdef unsigned long long u, u_o

    for i in (-9223372036854775808, -9223372036854775807, -1, 0, 1, 9223372036854775806, 9223372036854775807):
        x = pack_int64(i)
        y = struct.pack("<q", i)
        assert x == y, f"pack_int64 {x} not equal {y}"
        i_o = unpack_int64(x, 0)
        assert i == i_o, f"unpack_int64 {i_o} not equal {i}"

    print("Pass: pack_int64 & unpack_int64")

    for u in (0, 1, 18446744073709551613, 18446744073709551614, 18446744073709551615):
        x = pack_uint64(u)
        y = struct.pack("<Q", u)
        assert x == y, f"pack_uint64 {x} not equal {y}"
        u_o = unpack_uint64(x, 0)
        assert u == u_o, f"unpack_uint64 {u_o} not equal {u}"

    print("Pass: pack_uint64 & unpack_uint64")

    del struct

cpdef bint _test_gen_length_encoded_integer() except -1:
    cdef unsigned long long u

    for u in range(250):
        x = gen_length_encoded_integer(u)
        y = pack_uint8(u)
        assert x == y, f"gen_length_encoded_integer({u}): {x} not equal {y}"

    for u in (251, 65_535):
        x = gen_length_encoded_integer(u)
        y = pack_uint8(UNSIGNED_SHORT_COLUMN) + pack_uint16(u)
        assert x == y, f"gen_length_encoded_integer({u}): {x} not equal {y}"

    for u in (65_536, 16_777_215):
        x = gen_length_encoded_integer(u)
        y = pack_uint8(UNSIGNED_INT24_COLUMN) + pack_uint24(u)
        assert x == y, f"gen_length_encoded_integer({u}): {x} not equal {y}"

    for u in (16_777_216, 4_294_967_295):
        x = gen_length_encoded_integer(u)
        y = pack_uint8(UNSIGNED_INT64_COLUMN) + pack_uint64(u)
        assert x == y, f"gen_length_encoded_integer({u}): {x} not equal {y}"

    print("Pass: gen_length_encoded_integer")

cpdef bint _test_validate_max_allowed_packet() except -1:
    cdef int default = 16_777_216
    cdef int max_value = 1_073_741_824
    cdef int v
    cdef int i

    # Test None
    v = validate_max_allowed_packet(None, default, max_value)
    assert v == default, f"validate_max_allowed_packet: {v} not equal {default}"

    # Test integer
    for i in range(1, 256):
        v = validate_max_allowed_packet(i, default, max_value)
        assert v == i, f"validate_max_allowed_packet: {v} not equal {i}"

    for i in (0, max_value + 1):
        try:
            validate_max_allowed_packet(i, default, max_value)
        except errors.InvalidConnetionArgumentError as err:
            pass
        else:
            raise AssertionError("validate_max_allowed_packet ValueError is not raised")

    # Test string
    for val in ("1", "1b", "1B"):  # 1 byte
        v = validate_max_allowed_packet(val, default, max_value)
        assert v == 1, f"validate_max_allowed_packet: {v} not equal {1}"
    for val in ("1k", "1K", "1kb", "1KB"):
        v = validate_max_allowed_packet(val, default, max_value)
        assert v == 1_024, f"validate_max_allowed_packet: {v} not equal {1_024}"
    for val in ("1m", "1M", "1mb", "1MB"):
        v = validate_max_allowed_packet(val, default, max_value)
        assert v == 1_048_576, f"validate_max_allowed_packet: {v} not equal {1_048_576}"
    for val in ("1g", "1G", "1gb", "1GB"):
        v = validate_max_allowed_packet(val, default, max_value)
        assert v == 1_073_741_824, f"validate_max_allowed_packet: {v} not equal {1_073_741_824}"

    for val in (
        *["0" + sfix for sfix in ("b", "B", "k", "K", "kb", "KB", "m", "M", "mb", "MB", "g", "G", "gb", "GB")],
        *[str(max_value + 1) + sfix for sfix in ("b", "B")],
        *[str(int(max_value / 1024) + 1) + sfix for sfix in ("k", "K", "kb", "KB")],
        *[str(int(max_value / 1_048_576) + 1) + sfix for sfix in ("m", "M", "mb", "MB")],
        *[str(int(max_value / 1_073_741_824) + 1) + sfix for sfix in ("g", "G", "gb", "GB")],
        "3L",
    ):
        try:
            validate_max_allowed_packet(val, default, max_value)
        except errors.InvalidConnetionArgumentError as err:
            pass
        else:
            raise AssertionError("validate_max_allowed_packet ValueError is not raised")
