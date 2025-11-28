# cython: language_level=3

from libc.string cimport memchr as find_byte_in_memory
from cpython.bytes cimport (
    PyBytes_Size as bytes_len,
    PyBytes_AsString as bytes_to_chars,
    PyBytes_FromStringAndSize,
)
from cpython.unicode cimport (
    PyUnicode_Decode,
    PyUnicode_DecodeUTF8,
    PyUnicode_DecodeASCII,
    PyUnicode_DecodeLatin1,
    PyUnicode_AsEncodedString,
)
from sqlcycli.charset cimport Charset

# Constants
cdef:
    str DEFAULT_USER
    str DEFUALT_CHARSET
    int MAX_CONNECT_TIMEOUT
    int DEFALUT_MAX_ALLOWED_PACKET
    int MAXIMUM_MAX_ALLOWED_PACKET
    unsigned int MAX_PACKET_LENGTH
    #: Max statement size which :meth:`executemany` generates.
    #: Max size of allowed statement is max_allowed_packet - packet_header_size.
    #: Default value of max_allowed_packet is 1048576.
    unsigned int MAX_STATEMENT_LENGTH
    #: Regular expression for :meth:`Cursor.executemany`.
    #: executemany only supports simple bulk insert.
    #: You can use it to load large dataset.
    object INSERT_VALUES_RE
    #: Regular expression for server version.
    object SERVER_VERSION_RE
    # The following values are for the first byte
    # value of MySQL length encoded integer.
    unsigned char NULL_COLUMN  # 251
    unsigned char UNSIGNED_CHAR_COLUMN  # 251
    unsigned char UNSIGNED_SHORT_COLUMN  # 252
    unsigned char UNSIGNED_INT24_COLUMN  # 253
    unsigned char UNSIGNED_INT64_COLUMN  # 254

# Utils: string
cdef inline bytes encode_str(object data, const char* encoding):
    """Encode string to bytes using the 'encoding' with 'surrogateescape' error handling `<'bytes'>`.
    
    :param data `<'str'>`: String to encode.
    :param encoding `<'char*/bytes'>`: The encoding to use.
    :returns `<'bytes'>`: Encoded bytes.
    """
    return PyUnicode_AsEncodedString(data, encoding, b"surrogateescape")

cdef inline str decode_bytes(object data, const char* encoding):
    """Decode bytes to string using specified encoding with 'surrogateescape' error handling `<'str'>`.

    :param data `<'bytes'>`: Bytes to decode.
    :param encoding `<'char*/bytes'>`: Encoding to use for decoding.
    :returns `<'str'>`: Decoded string.
    """
    return PyUnicode_Decode(bytes_to_chars(data), bytes_len(data), encoding, b"surrogateescape")

cdef inline str decode_bytes_utf8(object data):
    """Decode bytes to string using 'utf-8' encoding with 'surrogateescape' error handling `<'str'>`.
    
    :param data `<'bytes'>`: Bytes to decode with `'utf-8'` encoding.
    :returns `<'str'>`: Decoded string.
    """
    return PyUnicode_DecodeUTF8(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str decode_bytes_ascii(object data):
    """Decode bytes to string using 'ascii' encoding with 'surrogateescape' error handling `<'str'>`.
    
    :param data `<'bytes'>`: Bytes to decode with `'ascii'` encoding.
    :returns `<'str'>`: Decoded string.
    """
    return PyUnicode_DecodeASCII(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str decode_bytes_latin1(object data):
    """Decode bytes to string using 'latin1' encoding with 'surrogateescape' error handling `<'str'>`.
    
    :param data `<'bytes'>`: Bytes to decode with `'latin1'` encoding.
    :returns `<'str'>`: Decoded string.
    """
    return PyUnicode_DecodeLatin1(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline Py_ssize_t find_null_byte(const char* data, Py_ssize_t length, Py_ssize_t pos) except -2:
    """Find the next NULL ('\\0') byte in a buffer starting at `pos` `<'int'>`.

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
    # Guard
    if pos < 0:
        raise ValueError("find_null_byte: argument 'pos' cannot be negative.")
    if length < 0:
        raise ValueError("find_null_byte: argument 'length' cannot be negative.")
    cdef Py_ssize_t remain = length - pos
    if remain <= 0:
        return -1

    # Find NULL byte
    cdef:
        const char* ptr = data + pos
        const char* loc = <const char*> find_byte_in_memory(ptr, 0, remain)

    if loc is NULL:
        return -1
    return <Py_ssize_t> (loc - data)

# Utils: Pack unsigned integer to bytes
cdef inline bytes pack_I24B(unsigned long long i, unsigned char b):
    """Pack a 24-bit little-endian unsigned integer and 
    a trailing byte into a 4-byte sequence `<'bytes'>`.

    :param i `<'int'>`: The unsigned integer; only least significant 
        24 bits are preserved (`0 <= i <= 16777215`).
    :param b `<'int'>`: The trailing byte (0 <= b <= 255).
    :returns `<'bytes'>`: The packed byte sequence.

    ## Equivalent
    >>> struct.pack("<I", i)[0:3] + struct.pack("<B", b)
    """
    cdef char buf[4]
    buf[0] =  i        & 0xFF
    buf[1] = (i >> 8)  & 0xFF
    buf[2] = (i >> 16) & 0xFF
    buf[3] =  b
    return PyBytes_FromStringAndSize(buf, 4)

cdef inline bytes pack_IB(unsigned long long i, unsigned char b):
    """Pack a 32-bit little-endian unsigned integer and a trailing byte into a 5-byte sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack as a 32-bit little-endian unsigned value.
    :param b `<'int'>`: The trailing byte (0 <= b <= 255).
    :returns `<'bytes'>`: The packed 5-byte sequence.

    ## Equivalent
    >>> struct.pack("<IB", i, b)
    """
    cdef char buf[5]
    buf[0] =  i        & 0xFF
    buf[1] = (i >> 8)  & 0xFF
    buf[2] = (i >> 16) & 0xFF
    buf[3] = (i >> 24) & 0xFF
    buf[4] =  b
    return PyBytes_FromStringAndSize(buf, 5)

cdef inline bytes pack_IIB23s(unsigned long long i1, unsigned long long i2, unsigned char b):
    """Pack two 32-bit little-endian unsigned integers, a byte, and 
    a 23-byte zero-filled string into a 32-byte sequence `<'bytes'>`.

    :param i1 `<'int'>`: The first 32-bit unsigned integer.
    :param i2 `<'int'>`: The second 32-bit unsigned integer.
    :param b `<'int'>`: The trailing byte (0 <= b <= 255).
    :returns `<'bytes'>`: The packed 32-byte sequence.

    ## Equivalent
    >>> struct.pack("<IIB23s", i1, i2, b, b"")
    """
    cdef Py_ssize_t idx
    cdef char buf[32]
    buf[0] =  i1        & 0xFF
    buf[1] = (i1 >> 8)  & 0xFF
    buf[2] = (i1 >> 16) & 0xFF
    buf[3] = (i1 >> 24) & 0xFF
    buf[4] =  i2        & 0xFF
    buf[5] = (i2 >> 8)  & 0xFF
    buf[6] = (i2 >> 16) & 0xFF
    buf[7] = (i2 >> 24) & 0xFF
    buf[8] =  b
    for idx in range(9, 32):
        buf[idx] = 0
    return PyBytes_FromStringAndSize(buf, 32)

cdef inline bytes pack_uint8(unsigned int i):
    """Pack an unsigned 8-bit integer into 1-byte sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant 
        8 bits are preserved (`0 <= i <= 255`).
    :returns `<'bytes'>`: A 1-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<B", i)
    """
    cdef char buf[1]
    buf[0] = i & 0xFF
    return PyBytes_FromStringAndSize(buf, 1)

cdef inline bytes pack_uint16(unsigned int i):
    """Pack an unsigned 16-bit integer into 2-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant 
        16 bits are preserved (`0 <= i <= 65535`).
    :returns `<'bytes'>`: A 2-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<H", i)
    """
    cdef char buf[2]
    buf[0] =  i       & 0xFF
    buf[1] = (i >> 8) & 0xFF
    return PyBytes_FromStringAndSize(buf, 2)

cdef inline bytes pack_uint24(unsigned long long i):
    """Pack an unsigned 24-bit integer into 3-byte little-endian sequence `<'bytes'>`.
    
    :param i `<'int'>`: The integer to pack; only least significant 
        24 bits are preserved (`0 <= i <= 16777215`).
    :returns `<'bytes'>`: A 3-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<I", i)[:3]
    """
    cdef char buf[3]
    buf[0] =  i        & 0xFF
    buf[1] = (i >> 8)  & 0xFF
    buf[2] = (i >> 16) & 0xFF
    return PyBytes_FromStringAndSize(buf, 3)

cdef inline bytes pack_uint32(unsigned long long i):
    """Pack an unsigned 32-bit integer into 4-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant 
        32 bits are preserved (`0 <= i <= 4294967295`).
    :returns `<'bytes'>`: A 4-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<I", i)
    """
    cdef char buf[4]
    buf[0] =  i        & 0xFF
    buf[1] = (i >> 8)  & 0xFF
    buf[2] = (i >> 16) & 0xFF
    buf[3] = (i >> 24) & 0xFF
    return PyBytes_FromStringAndSize(buf, 4)

cdef inline bytes pack_uint64(unsigned long long i):
    """Pack an unsigned 64-bit integer into 8-byte little-endian sequence `<'bytes'>`.
    
    :param i `<'int'>`: The integer to pack; only least significant 
        64 bits are preserved (`0 <= i <= 18446744073709551615`).
    :returns `<'bytes'>`: An 8-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<Q", i)
    """
    cdef char buf[8]
    buf[0] =  i        & 0xFF
    buf[1] = (i >> 8)  & 0xFF
    buf[2] = (i >> 16) & 0xFF
    buf[3] = (i >> 24) & 0xFF
    buf[4] = (i >> 32) & 0xFF
    buf[5] = (i >> 40) & 0xFF
    buf[6] = (i >> 48) & 0xFF
    buf[7] = (i >> 56) & 0xFF
    return PyBytes_FromStringAndSize(buf, 8)

cdef inline bytes gen_length_encoded_integer(unsigned long long i):
    """Generate a MySQL Protocol `LengthEncodedInteger` `<'bytes'>`.

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
    cdef char buf[9]
    #: Value 251 is reserved for NULL, so only 0-250, 252-254
    #: are used as the first byte of a length-encoded integer.
    if i < UNSIGNED_CHAR_COLUMN:        # 251
        buf[0] =  i        & 0xFF
        return PyBytes_FromStringAndSize(buf, 1)

    elif i < 1 << 16:
        buf[0] = UNSIGNED_SHORT_COLUMN  # 252
        buf[1] =  i        & 0xFF
        buf[2] = (i >> 8)  & 0xFF
        return PyBytes_FromStringAndSize(buf, 3)

    elif i < 1 << 24:
        buf[0] = UNSIGNED_INT24_COLUMN  # 253
        buf[1] =  i        & 0xFF
        buf[2] = (i >> 8)  & 0xFF
        buf[3] = (i >> 16) & 0xFF
        return PyBytes_FromStringAndSize(buf, 4)

    else:
        buf[0] = UNSIGNED_INT64_COLUMN  # 254
        buf[1] =  i        & 0xFF
        buf[2] = (i >> 8)  & 0xFF
        buf[3] = (i >> 16) & 0xFF
        buf[4] = (i >> 24) & 0xFF
        buf[5] = (i >> 32) & 0xFF
        buf[6] = (i >> 40) & 0xFF
        buf[7] = (i >> 48) & 0xFF
        buf[8] = (i >> 56) & 0xFF
        return PyBytes_FromStringAndSize(buf, 9)

# Utils: Pack integer to bytes
cdef inline bytes pack_int8(int i):
    """Pack a signed 8-bit integer into 1-byte sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant 
        8 bits are preserved (`-128 <= i <= 127`).
    :returns `<'bytes'>`: A 1-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<b", i)
    """
    cdef char buf[1]
    buf[0] = i & 0xFF
    return PyBytes_FromStringAndSize(buf, 1)

cdef inline bytes pack_int16(int i):
    """Pack a signed 16-bit integer into 2-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant 
        16 bits are preserved (`-32768 <= i <= 32767`).
    :returns `<'bytes'>`: A 2-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<h", i)
    """
    cdef char buf[2]
    buf[0] =  i       & 0xFF
    buf[1] = (i >> 8) & 0xFF
    return PyBytes_FromStringAndSize(buf, 2)

cdef inline bytes pack_int24(long long i):
    """Pack a signed 24-bit integer into 3-byte little-endian sequence `<'bytes'>`.
    
    :param i `<'int'>`: The integer to pack; only least significant 
        24 bits are preserved (`-8388608 <= i <= 8388607`).
    :returns `<'bytes'>`: A 3-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<i", i)[:3]
    """
    cdef char buf[3]
    buf[0] =  i        & 0xFF
    buf[1] = (i >> 8)  & 0xFF
    buf[2] = (i >> 16) & 0xFF
    return PyBytes_FromStringAndSize(buf, 3)

cdef inline bytes pack_int32(long long i):
    """Pack a signed 32-bit integer into 4-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant 
        32 bits are preserved (`-2147483648 <= i <= 2147483647`).
    :returns `<'bytes'>`: A 4-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<i", i)
    """
    cdef char buf[4]
    buf[0] =  i        & 0xFF
    buf[1] = (i >> 8)  & 0xFF
    buf[2] = (i >> 16) & 0xFF
    buf[3] = (i >> 24) & 0xFF
    return PyBytes_FromStringAndSize(buf, 4)

cdef inline bytes pack_int64(long long i):
    """Pack a signed 64-bit integer into 8-byte little-endian sequence `<'bytes'>`.

    :param i `<'int'>`: The integer to pack; only least significant 
        64 bits are preserved (`-9223372036854775808 <= i <= 9223372036854775807`).
    :returns `<'bytes'>`: An 8-byte sequence containing the packed value.

    ## Equivalent
    >>> struct.pack("<q", i)
    """
    cdef char buf[8]
    buf[0] =  i        & 0xFF
    buf[1] = (i >> 8)  & 0xFF
    buf[2] = (i >> 16) & 0xFF
    buf[3] = (i >> 24) & 0xFF
    buf[4] = (i >> 32) & 0xFF
    buf[5] = (i >> 40) & 0xFF
    buf[6] = (i >> 48) & 0xFF
    buf[7] = (i >> 56) & 0xFF
    return PyBytes_FromStringAndSize(buf, 8)

# Utils: Unpack unsigned integer
cdef inline unsigned char unpack_uint8(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack an unsigned 8-bit integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the byte is read from.
    :returns `<'int'>`: The unpacked unsigned 8-bit integer.
    """
    # Guard
    if pos < 0:
        return 0
    
    # Unpack
    return <unsigned char> data[pos]

cdef inline unsigned short unpack_uint16(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack an unsigned 16-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 1]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 16-bit integer starts.
    :returns `<'int'>`: The unpacked unsigned 16-bit integer.
    """
    # Guard
    if pos < 0:
        return 0

    # Unpack
    cdef:
        const unsigned char* p = <const unsigned char*> (data + pos)
        unsigned short v0 = p[0], v1 = p[1]
    return v0 | (v1 << 8)

cdef inline unsigned int unpack_uint24(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack an unsigned 24-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 2]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 24-bit integer starts.
    :returns `<'int'>`: The unpacked unsigned 24-bit integer.
    """
    # Guard
    if pos < 0:
        return 0

    # Unpack
    cdef:
        const unsigned char* p = <const unsigned char*> (data + pos)
        unsigned int v0 = p[0], v1 = p[1], v2 = p[2]
    return v0 | (v1 << 8) | (v2 << 16)

cdef inline unsigned int unpack_uint32(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack an unsigned 32-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 3]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 32-bit integer starts.
    :returns `<'int'>`: The unpacked unsigned 32-bit integer.
    """
    # Guard
    if pos < 0:
        return 0

    # Unpack
    cdef:
        const unsigned char* p = <const unsigned char*> (data + pos)
        unsigned int v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3]
    return v0 | (v1 << 8) | (v2 << 16) | (v3 << 24)

cdef inline unsigned long long unpack_uint64(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack an unsigned 64-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 7]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 64-bit integer starts.
    :returns `<'int'>`: The unpacked unsigned 64-bit integer.
    """
    # Guard
    if pos < 0:
        return 0

    # Unpack
    cdef:
        const unsigned char* p = <const unsigned char*> (data + pos)
        unsigned long long v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3]
        unsigned long long v4 = p[4], v5 = p[5], v6 = p[6], v7 = p[7]
    return v0 | (v1 << 8) | (v2 << 16) | (v3 << 24) | (v4 << 32) | (v5 << 40) | (v6 << 48) | (v7 << 56)

# Utils: Unpack signed integer
cdef inline char unpack_int8(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack a signed 8-bit integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'char*/bytes'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the byte is read from.
    :returns `<'int'>`: The unpacked signed 8-bit integer.
    """
    return <signed char> unpack_uint8(data, pos)

cdef inline short unpack_int16(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack a signed 16-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 1]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'const char*'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 16-bit integer starts.
    :returns `<'int'>`: The unpacked signed 16-bit integer.
    """
    return <signed short> unpack_uint16(data, pos)

cdef inline int unpack_int24(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack a signed 24-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 2]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'const char*'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 24-bit integer starts.
    :returns `<'int'>`: The unpacked signed 24-bit integer.
    """
    cdef int i = <signed int> unpack_uint24(data, pos)
    return i if i < 0x800000 else i - 0x1000000

cdef inline int unpack_int32(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack a signed 32-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 3]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'const char*'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 32-bit integer starts.
    :returns `<'int'>`: The unpacked signed 32-bit integer.
    """
    return <signed int> unpack_uint32(data, pos)

cdef inline long long unpack_int64(const char* data, Py_ssize_t pos) noexcept nogil:
    """Unpack a signed 64-bit little-endian integer from 'data' at offset 'pos'.

    ## Precondition
    - `pos` must be non-negative; otherwise return `0` directly.
    - `data[pos + 7]` must be within a valid readable buffer; 
        otherwise leading to undefined behavior.

    :param data `<'const char*'>`: Pointer to the byte buffer.
    :param pos  `<'int'>`: Zero-based offset where the 64-bit integer starts.
    :returns `<'int'>`: The unpacked signed 64-bit integer.
    """
    return <signed long long> unpack_uint64(data, pos)

# Utils: Query
cpdef str format_sql(str sql, object args)

# Utils: Connection
cpdef bytes gen_connect_attrs(list attrs)
cdef bytes DEFAULT_CONNECT_ATTRS

# Utils: Argument Validator
cpdef str validate_arg_str(object arg, str arg_name, str default)
cpdef object validate_arg_int(object arg, str arg_name, long long min_value, long long max_value)
cpdef bytes validate_arg_bytes(object arg, str arg_name, const char* encoding, str default)
cpdef Charset validate_charset(object charset, object collation, str default_charset)
cpdef int validate_autocommit(object autocommit) except -2
cpdef int validate_max_allowed_packet(object max_allowed_packet, int default, int max_value)
cpdef str validate_sql_mode(object sql_mode)
cpdef object validate_ssl(object ssl)
