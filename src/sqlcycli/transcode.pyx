# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
cimport cython
cimport numpy as np
from libc cimport math
from libc.limits cimport LLONG_MIN
from cpython cimport datetime
from cpython.set cimport PySet_Size as set_len
from cpython.dict cimport PyDict_Size as dict_len
from cpython.list cimport PyList_Size as list_len, PyList_GetItem as list_getitem
from cpython.tuple cimport PyTuple_Size as tuple_len, PyTuple_GetItem as tuple_getitem
from cpython.bytes cimport (
    PyBytes_Size as bytes_len,
    PyBytes_AsString as bytes_to_chars, 
    PyBytes_FromObject
)
from cpython.unicode cimport (
    PyUnicode_GET_LENGTH as str_len,
    PyUnicode_READ_CHAR as str_read,
    PyUnicode_Split as str_split,
    PyUnicode_Substring as str_substr,
    PyUnicode_Translate as str_translate,
    PyUnicode_Decode,
    PyUnicode_DecodeUTF8,
    PyUnicode_DecodeASCII,
)
from cytimes cimport utils as cyutils
from sqlcycli.constants cimport _FIELD_TYPE
from sqlcycli.sqlintvl cimport SQLInterval
from sqlcycli.sqlfunc cimport RawText, SQLFunction

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime
import numpy as np
import pandas as pd
from decimal import Decimal
from time import struct_time
from orjson import (
    loads as _orjson_loads, 
    dumps as _orjson_dumps, 
    OPT_SERIALIZE_NUMPY as _OPT_SERIALIZE_NUMPY
)
from cytimes import Pydt, Pddt
from sqlcycli import errors

# Type reference ======================================================================================
cdef:
    object T_DECIMAL = Decimal
    object T_STRUCT_TIME = struct_time
    object T_DICT_KEYS = type(dict().keys())
    object T_DICT_VALUES = type(dict().values())
    object T_DICT_ITEMS = type(dict().items())
    object T_NP_INT64 = np.int64
    object T_NP_INT32 = np.int32
    object T_NP_INT16 = np.int16
    object T_NP_INT8  = np.int8
    object T_NP_UINT64 = np.uint64
    object T_NP_UINT32 = np.uint32
    object T_NP_UINT16 = np.uint16
    object T_NP_UINT8 = np.uint8
    object T_NP_FLOAT64 = np.float64
    object T_NP_FLOAT32 = np.float32
    object T_NP_FLOAT16 = np.float16
    object T_NP_DATETIME64 = np.datetime64
    object T_NP_TIMEDELTA64 = np.timedelta64
    object T_NP_STR = np.str_
    object T_NP_BOOL = np.bool_
    object T_NP_BYTES = np.bytes_
    object T_NP_RECORD = np.record
    object T_PD_TIMESTAMP = pd.Timestamp
    object T_PD_TIMEDELTA = pd.Timedelta
    object T_PD_INDEX = pd.Index
    object T_PD_DATETIMEINDEX = pd.DatetimeIndex
    object T_PD_TIMEDELTAINDEX = pd.TimedeltaIndex
    object T_PD_SERIES = pd.Series
    object T_PD_DATAFRAME = pd.DataFrame
    object T_PD_NAT = pd.NaT.__class__
    object T_PYDT = Pydt
    object T_PDDT = Pddt

# Constants ===========================================================================================
#: table for string to literal translation
cdef list _ESCAPE_TABLE = [chr(x) for x in range(128)]
_ESCAPE_TABLE[0] = "\\0"
_ESCAPE_TABLE[ord("\\")] = "\\\\"
_ESCAPE_TABLE[ord("\n")] = "\\n"
_ESCAPE_TABLE[ord("\r")] = "\\r"
_ESCAPE_TABLE[ord("\032")] = "\\Z"
_ESCAPE_TABLE[ord('"')] = '\\"'
_ESCAPE_TABLE[ord("'")] = "\\'"
# . table to replace bracket '[]' to parenthesis '()'
cdef list _BRACKET_TABLE = [chr(x) for x in range(128)]
_BRACKET_TABLE[ord("[")] = "("
_BRACKET_TABLE[ord("]")] = ")"

# Utils ===============================================================================================
# . bytes
cdef inline str decode_bytes(object data, const char* encoding):
    """Decode bytes to string using specified encoding with 'surrogateescape' error handling `<'str'>`.

    :param data `<'bytes'>`: Bytes to decode.
    :param encoding `<'char*'>`: Encoding to use for decoding.
    :returns `<'str'>`: Decoded string.
    """
    return PyUnicode_Decode(bytes_to_chars(data), bytes_len(data), encoding, b"surrogateescape")

cdef inline str decode_bytes_utf8(object data):
    """Decode bytes to string using 'utf-8' encoding with 'surrogateescape' error handling `<'str'>`.
    
    :param data `<'bytes'>`: Bytes to decode.
    :returns `<'str'>`: Decoded string.
    """
    return PyUnicode_DecodeUTF8(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str decode_bytes_ascii(object data):
    """Decode bytes to string using 'ascii' encoding with 'surrogateescape' error handling `<'str'>`.
    
    :param data `<'bytes'>`: Bytes to decode.
    :returns `<'str'>`: Decoded string.
    """
    return PyUnicode_DecodeASCII(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str bytes_to_literal(object data):
    """Escape bytes object to string literal `<'bytes'>`.

    :param data `<'bytes'>`: Bytes object to escape.
    :returns `<'str'>`: Escaped string literal.

    ## Notes
    For more information, please refer to [PyMySQL](https://github.com/PyMySQL/PyMySQL)
    `converters.escape_bytes()` function.
    """
    return escape_str(decode_bytes_ascii(data))

cdef inline bint is_ascii_digit(char ch) noexcept nogil:
    """Check whether `ch` is an ASCII digit `<'bool'>`.

    - ASSCI digits: `'0'` (48) ... `'9'` (57)
    """
    return 48 <= ch <= 57

cdef inline unsigned long long unpack_uint_big_endian(const char* data, Py_ssize_t length, Py_ssize_t pos) noexcept nogil:
    """Unpack an unsigned big-endian integer from 'data' at offset 'pos'.

    :param data `<'char*'>`: Pointer to the data buffer.
    :param length `<'int'>`: Total length of the data buffer in bytes.
    :param pos `<'int'>`: Offset in the data buffer to start unpacking.
    :returns `<'int'>`: Unpacked unsigned big-endian integer.

    ## Precondition
    - `length` must be the actual total buffer length in bytes.
    - `data[0..length-1]` is a valid readable buffer.

    ## Behavior
    - If there are fewer than 1 byte available from `pos`, returns 0.
    - Uses up to 8 bytes starting at 'pos' (min(length - pos, 8)).
    - Interprets bytes in big-endian order: 
        * `data[pos]` is the most significant byte (MSB).
        * `data[pos + n - 1]` is the least significant byte (LSB), 
           where `n` is the number of bytes used (1 to 8).
    """
    # Guard
    if pos < 0:
        return 0
    cdef Py_ssize_t avail_bytes = length - pos
    if avail_bytes <= 0:
        return 0

    # Unpack
    cdef:
        const unsigned char* p = <const unsigned char*> (data + pos)
        unsigned long long c0, c1, c2, c3, c4, c5, c6, c7
    if avail_bytes == 1:
        c0 = p[0]
        return c0
    elif avail_bytes == 2:
        c0 = p[0]; c1 = p[1]
        return (c0 << 8) | c1
    elif avail_bytes == 3:
        c0 = p[0]; c1 = p[1]; c2 = p[2]
        return (c0 << 16) | (c1 << 8) | c2
    elif avail_bytes == 4:
        c0 = p[0]; c1 = p[1]; c2 = p[2]; c3 = p[3]
        return (c0 << 24) | (c1 << 16) | (c2 << 8) | c3
    elif avail_bytes == 5:
        c0 = p[0]; c1 = p[1]; c2 = p[2]; c3 = p[3]; c4 = p[4]
        return (c0 << 32) | (c1 << 24) | (c2 << 16) | (c3 << 8) | c4
    elif avail_bytes == 6:
        c0 = p[0]; c1 = p[1]; c2 = p[2]; c3 = p[3]; c4 = p[4]; c5 = p[5]
        return (c0 << 40) | (c1 << 32) | (c2 << 24) | (c3 << 16) | (c4 << 8) | c5
    elif avail_bytes == 7:
        c0 = p[0]; c1 = p[1]; c2 = p[2]; c3 = p[3]; c4 = p[4]; c5 = p[5]; c6 = p[6]
        return (c0 << 48) | (c1 << 40) | (c2 << 32) | (c3 << 24) | (c4 << 16) | (c5 << 8) | c6
    else:
        c0 = p[0]; c1 = p[1]; c2 = p[2]; c3 = p[3]; c4 = p[4]; c5 = p[5]; c6 = p[6]; c7 = p[7]
        return (c0 << 56) | (c1 << 48) | (c2 << 40) | (c3 << 32) | (c4 << 24) | (c5 << 16) | (c6 << 8) | c7

# . orjson
cdef inline object orjson_loads(object data):
    """Deserialize JSON string back to python object `<'Any'>`.

    :param data `<'bytes/str'>`: JSON string in bytes or str.
    :returns `<'Any'>`: Deserialized python object.

    ## Notes
    Based on [orjson](https://github.com/ijl/orjson) `loads()` function.
    """
    return _orjson_loads(data)

cdef inline str orjson_dumps(object obj):
    """Serialize python object to JSON string `<'str'>`.

    :param obj `<'Any'>`: Python object to serialize.
    :returns `<'str'>`: Serialized JSON string.

    ## Notes
    Based on [orjson](https://github.com/ijl/orjson) `dumps()` function.
    """
    return decode_bytes_utf8(_orjson_dumps(obj))

cdef inline str orjson_dumps_numpy(object obj):
    """Serialize `numpy.ndarray` to JSON string `<'str'>`.

    :param obj `<'numpy.ndarray'>`: Numpy array to serialize.
    :returns `<'str'>`: Serialized JSON string.

    ## Notes
    Based on [orjson](https://github.com/ijl/orjson) `dumps()` function.
    """
    return decode_bytes_utf8(_orjson_dumps(obj, option=_OPT_SERIALIZE_NUMPY))

# . numpy
cdef inline object arr_1d_getitem(np.ndarray arr, np.npy_intp i):
    """Get item from a 1-dimensional array `<'object'>`.
    
    :param arr `<'ndarray'>`: Numpy array.
    :param i `<'int'>`: Index of the item to get.
    :returns `<'object'>`: The item at the specified index.
    """
    cdef void* itemptr = <void*> np.PyArray_GETPTR1(arr, i)
    return np.PyArray_GETITEM(arr, itemptr)

cdef inline object arr_2d_getitem(np.ndarray arr, np.npy_intp i, np.npy_intp j):
    """Get item from a 2-dimensional array `<'object'>`.
    
    :param arr `<'ndarray'>`: Numpy array.
    :param i `<'int'>`: Row index of the item to get.
    :param j `<'int'>`: Column index of the item to get.
    :returns `<'object'>`: The item at the specified row and column.
    """
    cdef void* itemptr = <void*> np.PyArray_GETPTR2(arr, i, j)
    return np.PyArray_GETITEM(arr, itemptr)

cdef inline bint is_arr_1d_float_finite(np.ndarray arr, np.npy_intp size=-1) except -1:
    """Check if all elements in the 1-dimension float array is finite `<'bool'>`
    
    :param arr `<ndarray'>`: The 1-dimensional float array to check.
    :param size `<'int'>`: Optional size of the ndarray. Defaults to `-1`.
    :returns `<'bool'>`: True if all elements are finite; False otherwise.
    """
    # Setup
    cdef:
        int dtype = np.PyArray_TYPE(arr)
        np.npy_float64* float64_ptr
        np.npy_float32* float32_ptr
        np.npy_intp i
    if size < 0:
        size = arr.shape[0]

    # Check: float64
    if dtype == np.NPY_TYPES.NPY_FLOAT64:
        float64_ptr = <np.npy_float64*> np.PyArray_DATA(arr)
        for i in range(size):
            if not math.isfinite(float64_ptr[i]):
                return False
        return True

    # Cast: float16 -> float32
    if dtype == np.NPY_TYPES.NPY_FLOAT16:
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        dtype = np.NPY_TYPES.NPY_FLOAT32

    # Check: float32
    if dtype == np.NPY_TYPES.NPY_FLOAT32:
        float32_ptr = <np.npy_float32*> np.PyArray_DATA(arr)
        for i in range(size):
            if not math.isfinite(float32_ptr[i]):
                return False
        return True

    # Unsupported dtype
    raise AssertionError(
        "is_arr_1d_float_finite: The ndarray dtype must be 'float16', 'float32' or 'float64', "
        "instead got 'ndarray[%s]'." % arr.dtype
    )

cdef inline bint is_arr_2d_float_finite(np.ndarray arr, np.npy_intp size_i=-1, np.npy_intp size_j=-1) except -1:
    """Check if all elements in the 2-dimension float array is finite `<'bool'>`
    
    :param arr `<ndarray'>`: The 2-dimensional float array to check.
    :param size_i `<'int'>`: Optional size of the first dimension. Defaults to `-1`.
    :param size_j `<'int'>`: Optional size of the second dimension. Defaults to `-1`.
    :returns `<'bool'>`: True if all elements are finite; False otherwise.
    """
    # Setup
    cdef:
        int dtype = np.PyArray_TYPE(arr)
        np.npy_float64* float64_ptr
        np.npy_float32* float32_ptr
        np.npy_intp i, j, i_stride
    if size_i < 0:
        size_i = arr.shape[0]
    if size_j < 0:
        size_j = arr.shape[1]

    # Check: float64
    if dtype == np.NPY_TYPES.NPY_FLOAT64:
        float64_ptr = <np.npy_float64*> np.PyArray_DATA(arr)
        for i in range(size_i):
            i_stride = i * size_j
            for j in range(size_j):
                if not math.isfinite(float64_ptr[i_stride + j]):
                    return False
        return True

    # Cast: float16 -> float32
    if dtype == np.NPY_TYPES.NPY_FLOAT16:
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        dtype = np.NPY_TYPES.NPY_FLOAT32

    # Check: float32
    if dtype == np.NPY_TYPES.NPY_FLOAT32:
        float32_ptr = <np.npy_float32*> np.PyArray_DATA(arr)
        for i in range(size_i):
            i_stride = i * size_j
            for j in range(size_j):
                if not math.isfinite(float32_ptr[i_stride + j]):
                    return False
        return True

    # Unsupported dtype
    raise AssertionError(
        "is_arr_2d_float_finite: The ndarray dtype must be 'float16', 'float32' or 'float64', "
        "instead got 'ndarray[%s]'." % arr.dtype
    )

# Custom Escape Types ---------------------------------------------------------------------------------
cdef class ObjStr:
    """For any subclass of <'ObjStr'>, the `escape()` function will
    call its '__str__' method and use the result as the escaped value.

    The '__str__' method must be implemented in the subclass.
    """
    def __str__(self) -> str:
        raise NotImplementedError(
            "ObjStr subclass '%s' must implement it's own '__str__' "
            "method as how it should be escaped." % self.__class__.__name__
        )

cdef class CustomEscapeType(ObjStr):
    """The base class for custom escape types.

    To create a custom escape type, subclass this `CustomEscapeType` and implement 
    the `__str__` method to define how to escape the underlying object. The subclass 
    then act as a wrapper for the underlying object that needs custom escaping 
    behavior when passed to the `escape()` function. The underlying object can 
    be accessed through the `obj` property, once initialized.

    ## Example
    ```python
    def class MyCustomType(CustomEscapeType):
        def __str__(self) -> str:
            return str(self.obj).upper()
    ```
    """
    def __init__(self, object obj) -> None:
        """The base class for custom escape types.

        To create a custom escape type, subclass this `CustomEscapeType` and implement 
        the `__str__` method to define how to escape the underlying object. The subclass 
        then act as a wrapper for the underlying object that needs custom escaping 
        behavior when passed to the `escape()` function. The underlying object can 
        be accessed through the `obj` property, once initialized.

        :param obj `<'Any'>`: The underlying object to be escaped.

        ## Example
        ```python
        def class MyCustomType(CustomEscapeType):
            def __str__(self) -> str:
                return str(self.obj).upper()
        ```
        """
        self._obj = obj

    @property
    def obj(self) -> object:
        """Returns the underlying object to be escaped `<'Any'>`."""
        return self._obj

    def __repr__(self) -> str:
        return "<'%s' (obj=%r)>" % (self.__class__.__name__, self._obj)

    def __str__(self) -> str:
        raise NotImplementedError(
            "CustomEscapeType subclass '%s' must implement it's own '__str__' "
            "method as how to escape the underlying object '%s' %s." 
            % (self.__class__.__name__, self._obj, type(self._obj))
        )

cdef class BIT(CustomEscapeType):
    """Represents a value of MySQL BIT column. Act as a wrapper
    for the BIT `obj`, so the `escape()` function can identify 
    and escape the value to the desired literal format.

    - Accepts raw bytes or integer value.
    - Validation & conversion only happens when processed by the `escape()` function.
    """
    def __str__(self) -> str:
        cdef object obj = self._obj

        # Exact type
        dtype = type(obj)
        # . bytes
        if dtype is bytes:
            return escape_int(_decode_bit(obj, True))
        # . bytes-like
        if dtype is T_NP_BYTES or dtype is bytearray or dtype is memoryview:
            return escape_int(_decode_bit(PyBytes_FromObject(obj), True))
        # . integer
        if dtype is int:
            return escape_int(obj)
        # . integer-like
        if (
            dtype is T_NP_INT64 or dtype is T_NP_INT32 or dtype is T_NP_INT16 or dtype is T_NP_INT8 or
            dtype is T_NP_UINT64 or dtype is T_NP_UINT32 or dtype is T_NP_UINT16 or dtype is T_NP_UINT8
        ):
            return escape_int(obj)

        # Subclass
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return escape_int(_decode_bit(PyBytes_FromObject(obj), True))
        if isinstance(obj, int):
            return escape_int(obj)

        # Unsupported type
        raise TypeError(
            "Cannot escape BIT value from %s. "
            "Only supports 'bytes', 'bytearray', 'memoryview' and 'int'." 
            % type(obj)
        )

cdef class JSON(CustomEscapeType):
    """Represents a value for MySQL JSON column. Act as a wrapper
    for the JSON `obj`, so the `escape()` function can identify and
    escape the value to the desired literal format.

    - Accepts any objects that can be serialized to JSON format.
    - Do `NOT` pass already serialized JSON string to this class.
    - Validation & conversion only happens when processed by the `escape()` function.
    """
    def __str__(self) -> str:
        cdef object obj  = self._obj
        cdef bytes json
        try:
            json = _orjson_dumps(obj, option=_OPT_SERIALIZE_NUMPY)
        except Exception as err:
            raise ValueError(
                "Cannot escape JSON value from %s.\n"
                "%s" % (type(obj), err)
            ) from err
        return bytes_to_literal(json)

cdef inline str _escape_rawtext(RawText data):
    """Escape RawText instance to literal `<'str'>`.

    :param data `<'RawText'>`: RawText instance to escape.
    :returns `<'str'>`: Escaped SQL literal.
    """
    return data._value

cdef inline str _escape_sqlfunction(SQLFunction data):
    """Escape SQLFunction instance to literal `<'str'>`.

    :param data `<'SQLFunction'>`: SQLFunction instance to escape.
    :returns `<'str'>`: Escaped SQL literal.
    """
    cdef str syntax = data.syntax()
    if data._arg_count <= 0:
        return syntax
    elif data._arg_count == 1:
        return syntax % escape_common(<object> tuple_getitem(data._args, 0))
    else:
        return syntax % escape_tuple_items(data._args, False)

cdef inline str _escape_sqlinterval(SQLInterval data):
    """Escape SQLInterval instance to literal `<'str'>`.

    :param data `<'SQLInterval'>`: SQLInterval instance to escape.
    :returns `<'str'>`: Escaped SQL literal.
    """
    return data.syntax() % escape_common(data._expr)
    
# Escape ==============================================================================================
# . basic
cdef inline str escape_bool(object data):
    """Escape boolean to literal `<'str'>`.

    :param data `<'bool'>`: Boolean data to escape.
    :returns `<'str'>`: Escaped boolean literal: True -> `'1'`; False -> `'0'`
    """
    return "1" if bool(data) else "0"

cdef inline str escape_int(object data):
    """Escape integer to literal `<'str'>`.

    :param data `<'int'>`: Integer data to escape.
    :returns `<'str'>`: Escaped integer literal.
    """
    return str(data)

cdef inline str escape_float(object data):
    """Escape float to literal `<'str'>`.

    :param data `<'float'>`: Float data to escape.
    :returns `<'str'>`: Escaped float literal.
    :raises `<'ValueError'>`: If float value is invalid.
    """
    #: For normal float numbers, orjson performs
    #: faster than Python built-in `str()` function.
    if math.isnormal(data):
        return orjson_dumps(data)
    #: For other float objects, fallback to Python
    #: built-in `str()` approach.
    else:
        return escape_float64(data)

cdef inline str escape_str(object data):
    """Escape string to literal `<'str'>`.

    :param data `<'str'>`: String data to escape.
    :returns `<'str'>`: Escaped string literal.
    """
    return "'" + str_translate(data, _ESCAPE_TABLE, NULL) + "'"

cdef inline str escape_none(object data):
    """Escape None to literal `<'str'>`.

    :param data `<'NoneType'>`: NoneType data to escape.
    :returns `<'str'>`: Escaped NoneType literal: `NULL`.
    """
    return "NULL"

# . date&time
cdef inline str escape_datetime(object data):
    """Escape datetime to literal `<'str'>`.

    :param data `<'datetime.datetime'>`: Datetime data to escape.
    :returns `<'str'>`: Escaped datetime literal.
    """
    cdef int microsecond = datetime.datetime_microsecond(data)
    if microsecond == 0:
        return "'%04d-%02d-%02d %02d:%02d:%02d'" % (
            datetime.datetime_year(data),
            datetime.datetime_month(data),
            datetime.datetime_day(data),
            datetime.datetime_hour(data),
            datetime.datetime_minute(data),
            datetime.datetime_second(data),
        )
    else:
        return "'%04d-%02d-%02d %02d:%02d:%02d.%06d'" % (
            datetime.datetime_year(data),
            datetime.datetime_month(data),
            datetime.datetime_day(data),
            datetime.datetime_hour(data),
            datetime.datetime_minute(data),
            datetime.datetime_second(data),
            microsecond,
        )

cdef inline str escape_date(object data):
    """Escape date to literal `<'str'>`.

    :param data `<'datetime.date'>`: Date data to escape.
    :returns `<'str'>`: Escaped date literal.
    """
    return "'%04d-%02d-%02d'" % (
        datetime.date_year(data),
        datetime.date_month(data),
        datetime.date_day(data),
    )

cdef inline str escape_time(object data):
    """Escape time to literal `<'str'>`.

    :param data `<'datetime.time'>`: Time data to escape.
    :returns `<'str'>`: Escaped time literal.
    """
    cdef int microsecond = datetime.time_microsecond(data)
    if microsecond == 0:
        return "'%02d:%02d:%02d'" % (
            datetime.time_hour(data),
            datetime.time_minute(data),
            datetime.time_second(data),
        )
    else:
        return "'%02d:%02d:%02d.%06d'" % (
            datetime.time_hour(data),
            datetime.time_minute(data),
            datetime.time_second(data),
            microsecond,
        )

cdef inline str escape_timedelta(object data):
    """Escape timedelta to literal `<'str'>`.

    :param data `<'datetime.timedelta'>`: Timedelta data to escape.
    :returns `<'str'>`: Escaped timedelta literal.
    """
    cdef:
        long long days = datetime.timedelta_days(data)
        long long seconds = datetime.timedelta_seconds(data)
        long long us = datetime.timedelta_microseconds(data)
        long long hours, minutes, r
    seconds = days * 86400 + seconds

    # Positive timedelta
    if seconds >= 0:
        with cython.cdivision(True):
            hours   = seconds / 3600; r = seconds % 3600
            minutes = r / 60
            seconds = r % 60
        if us == 0:
            return "'%02d:%02d:%02d'" % (hours, minutes, seconds)
        else:
            return "'%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us)

    # Negative timedelta (microseconds == 0)
    elif us == 0:
        seconds = -seconds
        with cython.cdivision(True):
            hours   = seconds / 3600; r = seconds % 3600
            minutes = r / 60
            seconds = r % 60
        return "'-%02d:%02d:%02d'" % (hours, minutes, seconds)

    # Negative timedelta (microseconds != 0)
    else:
        us = -(seconds * cyutils.US_SECOND + us)
        with cython.cdivision(True):
            hours   = us / cyutils.US_HOUR;   r  = us % cyutils.US_HOUR
            minutes = r / cyutils.US_MINUTE;  r  = r % cyutils.US_MINUTE
            seconds = r / cyutils.US_SECOND;  us = r % cyutils.US_SECOND
        return "'-%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us)

cdef inline str escape_struct_time(object data):
    """Escape struct_time to literal `<'str'>`.

    :param data `<'time.struct_time'>`: Struct_time data to escape.
    :returns `<'str'>`: Escaped struct_time literal.
    """
    return "'%04d-%02d-%02d %02d:%02d:%02d'" % (
        data.tm_year, data.tm_mon, data.tm_mday,
        data.tm_hour, data.tm_min, data.tm_sec,
    )

# . bytes
cdef inline str escape_bytes(object data):
    """Escape bytes to literal `<'str'>`.

    :param data `<'bytes'>`: Bytes data to escape.
    :returns `<'str'>`: Escaped bytes literal.
    """
    return "_binary'" + str_translate(decode_bytes_ascii(data), _ESCAPE_TABLE, NULL) + "'"

cdef inline str escape_bytearray(object data):
    """Escape bytearray to literal `<'str'>`.

    :param data `<'bytearray'>`: Bytearray data to escape.
    :returns `<'str'>`: Escaped bytearray literal.
    """
    return escape_bytes(PyBytes_FromObject(data))

cdef inline str escape_memoryview(memoryview data):
    """Escape memoryview to literal `<'str'>`.

    :param data `<'memoryview'>`: Memoryview data to escape.
    :returns `<'str'>`: Escaped memoryview literal.
    """
    return escape_bytes(data.tobytes())

# . numeric
cdef inline str escape_decimal(object data):
    """Escape decimal to literal `<'str'>`.

    :param data `<'decimal.Decimal'>`: Decimal data to escape.
    :returns `<'str'>`: Escaped decimal literal.
    """
    cdef str res = str(data)
    cdef Py_UCS4 ch0 = str_read(res, 0)
    if ch0 in ("N", "n", "I", "i"):
        raise ValueError("The decimal value must be finite, instead got '%s'." % data)
    if ch0 in ("-", "+") and str_read(res, 1) in ("N", "n", "I", "i"):
        raise ValueError("The decimal value must be finite, instead got '%s'." % data)
    return res

# . sequence
cdef inline str escape_list(list data):
    """Escape list to literal `<'str'>`.

    :param data `<'list'>`: List data to escape.
    :returns `<'str'>`: Escaped list literal.
    """
    if list_len(data) == 0:
        return "()"  # exit

    cdef str res = ",".join([escape_common(i) for i in data])
    #: If the result already starts with '(', return as-is.
    #: "('a',1),('b',2)"
    if str_read(res, 0) == "(":
        return res
    #: Otherwise, wrap with parenthesis.
    #: "'a',1" -> "('a',1)"
    return "(" + res + ")"

cdef inline str escape_tuple(tuple data):
    """Escape tuple to literal `<'str'>`.

    :param data `<'tuple'>`: Tuple data to escape.
    :returns `<'str'>`: Escaped tuple literal.
    """
    if tuple_len(data) == 0:
        return "()"  # exit

    cdef str res = ",".join([escape_common(i) for i in data])
    #: If the result already starts with '(', return as-is.
    #: "('a',1),('b',2)"
    if str_read(res, 0) == "(":
        return res
    #: Otherwise, wrap with parenthesis.
    #: "'a',1" -> "('a',1)"
    return "(" + res + ")"

cdef inline str escape_set(set data):
    """Escape set to literal `<'str'>`.

    :param data `<'set'>`: Set data to escape.
    :returns `<'str'>`: Escaped set literal.
    """
    if set_len(data) == 0:
        return "()"  # exit

    cdef str res = ",".join([escape_common(i) for i in data])
    #: If the result already starts with '(', return as-is.
    #: "('a',1),('b',2)"
    if str_read(res, 0) == "(":
        return res
    #: Otherwise, wrap with parenthesis.
    #: "'a',1" -> "('a',1)"
    return "(" + res + ")"

cdef inline str escape_frozenset(frozenset data):
    """Escape frozenset to literal `<'str'>`.

    :param data `<'frozenset'>`: Frozenset data to escape.
    :returns `<'str'>`: Escaped frozenset literal.
    """
    if set_len(data) == 0:
        return "()"  # exit

    cdef str res = ",".join([escape_common(i) for i in data])
    #: If the result already starts with '(', return as-is.
    #: "('a',1),('b',2)"
    if str_read(res, 0) == "(":
        return res
    #: Otherwise, wrap with parenthesis.
    #: "'a',1" -> "('a',1)"
    return "(" + res + ")"

cdef inline str escape_range(object data):
    """Escape range to literal `<'str'>`.

    :param data `<'range'>`: Range data to escape.
    :returns `<'str'>`: Escaped range literal.
    """
    return "(" + ",".join([escape_common(i) for i in data]) + ")"

cdef inline str escape_sequence(object data):
    """Escape sequence (list, tuple, set, frozenset, range) to literal `<'str'>`.

    :param data `<'sequence'>`: Sequence data to escape.
    :returns `<'str'>`: Escaped sequence literal.
    """
    cdef str res = ",".join([escape_common(i) for i in data])
    if str_len(res) == 0:
        return "()"  # exit

    #: If the result already starts with '(', return as-is.
    #: "('a',1),('b',2)"
    if str_read(res, 0) == "(":
        return res
    #: Otherwise, wrap with parenthesis.
    #: "'a',1" -> "('a',1)"
    return "(" + res + ")"

# . mapping
cdef inline str escape_dict(dict data):
    """Escape dict to literal `<'str'>`.

    :param data `<'dict'>`: Dict data to escape.
    :returns `<'str'>`: Escaped dict literal.
    """
    if dict_len(data) == 0:
        return "()"  # exit

    cdef str res = ",".join([escape_common(i) for i in data.values()])
    #: If the result already starts with '(', return as-is.
    #: "('a',1),('b',2)"
    if str_read(res, 0) == "(":
        return res
    #: Otherwise, wrap with parenthesis.
    #: "'a',1" -> "('a',1)"
    return "(" + res + ")"

# . numpy
cdef inline str escape_float64(object data):
    """Escape numpy.float_ to literal `<'str'>`.

    :param data `<'numpy.float_'>`: Numpy float data to escape.
    :returns `<'str'>`: Escaped float literal.
    """
    #: For numpy.float64, Python built-in `str()` function 
    #: performs faster than orjson for most small float 
    #: numbers (with less than 6 decimal places).
    return str(data) if math.isfinite(data) else "NULL"

cdef inline str escape_datetime64(object data):
    """Escape numpy.datetime64 to literal `<'str'>`.

    :param data `<'numpy.datetime64'>`: Numpy datetime64 data to escape.
    :returns `<'str'>`: Escaped datetime64 literal.
    """
    return _escape_datetime64_value(np.get_datetime64_value(data), np.get_datetime64_unit(data))

cdef inline str _escape_datetime64_value(long long value, int unit):
    """(internal) Escape numpy.datetime64 integer value to literal `<'str'>`.

    :param value `<'int'>`: Numpy datetime64 integer value to escape.
    :param unit `<'int'>`: Numpy datetime64 unit (`ENUM NPY_DATETIMEUNIT`).
    :returns `<'str'>`: Escaped datetime64 literal.
    """
    # Handle NaT
    if value == LLONG_MIN:
        return "NULL"

    # Normalize to microseconds
    cdef long long us, r
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        with cython.cdivision(True):
            us = value / 1_000; r = value % 1_000
            if r < 0: us -= 1
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        us = value
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        us = value * cyutils.US_MILLISECOND
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        us = value * cyutils.US_SECOND
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        us = value * cyutils.US_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        us = value * cyutils.US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        us = value * cyutils.US_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        us = value * cyutils.US_DAY * 7
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        us = cyutils._dt64_M_as_int64_D(value, cyutils.US_DAY, 0)
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        us = cyutils._dt64_Y_as_int64_D(value, cyutils.US_DAY, 0)
    else:
        try:
            unit_str = cyutils.nptime_unit_int2str(unit)
        except Exception as err:
            raise ValueError("Cannot escape 'datetime64', time unit '%d' is not supported." % unit) from err
        else:
            raise ValueError("Cannot escape 'datetime64', time unit '%s' is not supported." % unit_str)

    # Escape to literal
    cdef cyutils.dtm _dtm = cyutils.dtm_fr_us(us)
    us = _dtm.microsecond
    if us == 0:
        return "'%04d-%02d-%02d %02d:%02d:%02d'" % (
            _dtm.year, _dtm.month, _dtm.day,
            _dtm.hour, _dtm.minute, _dtm.second,
        )
    else:
        return "'%04d-%02d-%02d %02d:%02d:%02d.%06d'" % (
            _dtm.year, _dtm.month, _dtm.day,
            _dtm.hour, _dtm.minute, _dtm.second, us,
        )

cdef inline str escape_timedelta64(object data):
    """Escape numpy.timedelta64 object to literal `<'str'>`.

    :param data `<'numpy.timedelta64'>`: Numpy timedelta64 data to escape.
    :returns `<'str'>`: Escaped timedelta64 literal.
    """
    return _escape_timedelta64_value(np.get_timedelta64_value(data), np.get_datetime64_unit(data))

cdef inline str _escape_timedelta64_value(long long value, int unit):
    """(internal) Escape numpy.timedelta64 integer value to literal `<'str'>`.

    :param value `<'int'>`: Numpy timedelta64 in microseconds to escape.
    :param unit `<'int'>`: Numpy timedelta64 unit (`ENUM NPY_DATETIMEUNIT`).
    :returns `<'str'>`: Escaped timedelta64 literal.
    """
    # Normalize to microseconds
    cdef long long us, r
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:   # nanosecond
        with cython.cdivision(True):
            us = value / 1_000; r = value % 1_000
            if r < 0: us -= 1
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:   # microsecond
        us = value 
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:   # millisecond
        us = value * cyutils.US_MILLISECOND
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:    # second
        us = value * cyutils.US_SECOND
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:    # minute
        us = value * cyutils.US_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:    # hour
        us = value * cyutils.US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:    # day
        us = value * cyutils.US_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_W:    # week
        us = value * cyutils.US_DAY * 7
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_M:    # month
        us = cyutils._td64_M_as_int64_D(value, np.NPY_DATETIMEUNIT.NPY_FR_us, 0)
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:    # year
        us = cyutils._td64_Y_as_int64_D(value, np.NPY_DATETIMEUNIT.NPY_FR_us, 0)
    else:
        try:
            unit_str = cyutils.nptime_unit_int2str(unit)
        except Exception as err:
            raise ValueError("Cannot escape 'timedelta64', time unit '%d' is not supported." % unit) from err
        else:
            raise ValueError("Cannot escape 'timedelta64', time unit '%s' is not supported." % unit_str)

    # Escape to literal
    cdef long long hours, minutes, seconds
    cdef bint neg
    if us < 0:
        neg = True; us = -us
    else:
        neg = False
    with cython.cdivision(True):
        hours   = us / cyutils.US_HOUR;   r  = us % cyutils.US_HOUR
        minutes = r / cyutils.US_MINUTE;  r  = r % cyutils.US_MINUTE
        seconds = r / cyutils.US_SECOND;  us = r % cyutils.US_SECOND
    if neg:
        if us == 0:
            return "'-%02d:%02d:%02d'" % (hours, minutes, seconds)
        return "'-%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us)
    else:
        if us == 0:
            return "'%02d:%02d:%02d'" % (hours, minutes, seconds)
        return "'%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us)

# . ndarray
cdef inline str escape_ndarray(np.ndarray data):
    """Escape numpy.ndarray to literal `<'str'>`.

    :param data `<'ndarray'>`: Numpy array data to escape.
    :returns `<'str'>`: Escaped ndarray literal.

    ## Example (1-dimension):
    >>> escape_ndarray(np.array([1, 2, 3], dtype=np.int64))
    >>> "(1,2,3)" 

    ## Example (2-dimension):
    >>> escape_ndarray(np.array(
        [[1, 2, 3], [3, 4, 5]], dtype=np.int64))
    >>> "(1,2,3),(3,4,5)"
    """
    # Get ndarray dtype
    cdef int dtype = np.PyArray_TYPE(data)

    # Escape
    # . ndarray[object]
    if dtype == np.NPY_TYPES.NPY_OBJECT:
        return _escape_ndarray_object(data)
    # . ndarray[int] & ndarray[uint]
    if dtype in (
        np.NPY_TYPES.NPY_INT64, np.NPY_TYPES.NPY_INT32,
        np.NPY_TYPES.NPY_INT16, np.NPY_TYPES.NPY_INT8,
        np.NPY_TYPES.NPY_UINT64, np.NPY_TYPES.NPY_UINT32,
        np.NPY_TYPES.NPY_UINT16, np.NPY_TYPES.NPY_UINT8,
    ):
        return _escape_ndarray_int(data)
    # . ndarray[float]
    if dtype in (
        np.NPY_TYPES.NPY_FLOAT64,
        np.NPY_TYPES.NPY_FLOAT32,
        np.NPY_TYPES.NPY_FLOAT16,
    ):
        return _escape_ndarray_float(data)
    # . ndarray[bool]
    if dtype == np.NPY_TYPES.NPY_BOOL:
        return _escape_ndarray_bool(data)
    # . ndarray[datetime64]
    if dtype == np.NPY_TYPES.NPY_DATETIME:
        return _escape_ndarray_dt64(data)
    # . ndarray[timedelta64]
    if dtype == np.NPY_TYPES.NPY_TIMEDELTA:
        return _escape_ndarray_td64(data)
    # . ndarray[bytes]
    if dtype == np.NPY_TYPES.NPY_STRING:
        return _escape_ndarray_bytes(data)
    # . ndarray[unicode]
    if dtype == np.NPY_TYPES.NPY_UNICODE:
        return _escape_ndarray_unicode(data)

    # Unsupported dtype
    raise TypeError("Cannot escape 'ndarray[%s]', array dtype is not supported." % data.dtype)
    
cdef inline str _escape_ndarray_object(np.ndarray arr):
    """(internal) Escape numpy.ndarray with `object` dtype to literal `<'str'>`.

    :param arr `<'ndarray[object]'>`: Numpy array in `object` dtype to escape.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return "()"  # exit
        res_l = [escape_common(arr_1d_getitem(arr, i)) for i in range(size_i)]
        return "(" + ",".join(res_l) + ")"

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return "()"  # exit
        size_i = shape[0]
        res_l = [
            "(" + ",".join([escape_common(arr_2d_getitem(arr, i, j)) for j in range(size_j)]) + ")"
            for i in range(size_i)
        ]
        return ",".join(res_l)

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline str _escape_ndarray_int(np.ndarray arr):
    """(internal) Escape numpy.ndarray with `integer` dtype to literal `<'str'>`.

    Supported dtypes: 
    - `int8`, `int16`, `int32`, `int64`
    - `uint8`, `uint16`, `uint32`, `uint64`

    :param arr `<'ndarray[int]'>`: Numpy array in `integer` dtype to escape.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        str res_s

    # 1-dimension
    if ndim == 1:
        if shape[0] == 0:
            return "()"  # exit
        #: 'res_s' will be like
        #: "[1,2,3]"
        res_s = orjson_dumps_numpy(arr)
        #: replace brackets with parenthesis
        #: "(1,2,3)"
        return str_translate(res_s, _BRACKET_TABLE, NULL)

    # 2-dimension
    if ndim == 2:
        if shape[1] == 0:
            return "()"  # exit
        #: 'res_s' will be like
        #: "[[1,2,3],[4,5,6]]"
        res_s = orjson_dumps_numpy(arr)
        #: remove the outer brackets
        #: "[1,2,3],[4,5,6]"
        res_s = str_substr(res_s, 1, str_len(res_s) - 1)
        #: replace brackets with parenthesis
        #: "(1,2,3),(4,5,6)"
        return str_translate(res_s, _BRACKET_TABLE, NULL)

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline str _escape_ndarray_float(np.ndarray arr):
    """(internal) Escape numpy.ndarray with `float` dtype to literal `<'str'>`.

    Supported dtypes: 
    - `float16`, `float32`, `float64`

    :param arr `<'ndarray[float]'>`: Numpy array in `float` dtype to escape.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j
        str  res_s
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return "()"  # exit
        # . slow approach for nan & inf
        if not is_arr_1d_float_finite(arr, size_i):
            res_l = _escape_ndarray_1d_float_items_slow(arr, size_i)
            return "(" + ",".join(res_l) + ")"
        #: 'res_s' will be like
        #: "[1.0,2.0,3.0]"
        res_s = orjson_dumps_numpy(arr)
        #: replace brackets with parenthesis
        #: "(1.0,2.0,3.0)"
        return str_translate(res_s, _BRACKET_TABLE, NULL)

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return "()"  # exit
        size_i = shape[0]
        # . slow approach for nan & inf
        if not is_arr_2d_float_finite(arr, size_i, size_j):
            return _escape_ndarray_2d_float_slow(arr, size_i, size_j)
        #: 'res_s' will be like
        #: "[[1.0,2.0,3.0],[4.0,5.0,6.0]]"
        res_s = orjson_dumps_numpy(arr)
        #: remove the outer brackets
        #: "[1.0,2.0,3.0],[4.0,5.0,6.0]"
        res_s = str_substr(res_s, 1, str_len(res_s) - 1)
        #: replace brackets with parenthesis
        #: "(1.0,2.0,3.0),(4.0,5.0,6.0)"
        return str_translate(res_s, _BRACKET_TABLE, NULL)

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline str _escape_ndarray_2d_float_slow(np.ndarray arr, np.npy_intp size_i=-1, np.npy_intp size_j=-1):
    """(internal) Escape 2-D numpy.ndarray with `float` dtype to 
    literal through native str (slow) approach `<'str'>`.
    
    :param arr `<ndarray'>`: The 2-dimensional float array to escape.
    :param size_i `<'int'>`: Optional size of the first dimension. Defaults to `-1`.
    :param size_j `<'int'>`: Optional size of the second dimension. Defaults to `-1`.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    # Setup
    cdef:
        int dtype = np.PyArray_TYPE(arr)
        np.npy_float64* f64_ptr
        np.npy_float64  f64_v
        np.npy_float32* f32_ptr
        np.npy_float32  f32_v
        np.npy_intp     i, j, i_stride
        list res_l, row_l
    if size_i < 0:
        size_i = arr.shape[0]
    if size_j < 0:
        size_j = arr.shape[1]

    # Escape: float64
    if dtype == np.NPY_TYPES.NPY_FLOAT64:
        f64_ptr = <np.npy_float64*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            row_l = []
            for j in range(size_j):
                f64_v = f64_ptr[i_stride + j]
                if not math.isfinite(f64_v):
                    row_l.append("NULL")
                else:
                    row_l.append(str(f64_v))
            res_l.append("(" + ",".join(row_l) + ")")
        return ",".join(res_l)

    # Cast: float16 -> float32
    if dtype == np.NPY_TYPES.NPY_FLOAT16:
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        dtype = np.NPY_TYPES.NPY_FLOAT32

    # Escape: float32
    if dtype == np.NPY_TYPES.NPY_FLOAT32:
        f32_ptr = <np.npy_float32*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            row_l = []
            for j in range(size_j):
                f32_v = f32_ptr[i_stride + j]
                if not math.isfinite(f32_v):
                    row_l.append("NULL")
                else:
                    row_l.append(str(f32_v))
            res_l.append("(" + ",".join(row_l) + ")")
        return ",".join(res_l)

    # Unsupported dtype
    raise AssertionError(
        "_escape_ndarray_2d_float_items_slow: The ndarray dtype must be 'float16', 'float32' or 'float64', "
        "instead got 'ndarray[%s]'." % arr.dtype
    )

cdef inline str _escape_ndarray_bool(np.ndarray arr):
    """(internal) Escape numpy.ndarray with `bool` dtype to literal `<'str'>`.

    :param arr `<'ndarray[bool]'>`: Numpy array in `bool` dtype to escape.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j, i_stride
        np.npy_bool* arr_ptr
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return "()"  # exit
        arr_ptr = <np.npy_bool*> np.PyArray_DATA(arr)
        res_l = ["1" if arr_ptr[i] else "0" for i in range(size_i)]
        return "(" + ",".join(res_l) + ")"

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return "()"  # exit
        size_i = shape[0]
        arr_ptr = <np.npy_bool*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            res_l.append(
                "(" + ",".join(
                    ["1" if arr_ptr[i_stride + j] else "0" 
                    for j in range(size_j)]
                ) + ")"
            )
        return ",".join(res_l)

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline str _escape_ndarray_dt64(np.ndarray arr):
    """(internal) Escape numpy.ndarray with `datetime64` dtype to literal `<'str'>`.

    :param arr `<'ndarray[datetime64]'>`: Numpy array in `datetime64` dtype to escape.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j, i_stride
        int unit
        np.npy_int64* arr_ptr
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return "()"  # exit
        unit = cyutils.get_arr_nptime_unit(arr)
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        res_l = [_escape_datetime64_value(arr_ptr[i], unit) for i in range(size_i)]
        return "(" + ",".join(res_l) + ")"

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return "()"  # exit
        size_i = shape[0]
        unit = cyutils.get_arr_nptime_unit(arr)
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            res_l.append(
                "(" + ",".join(
                    [_escape_datetime64_value(arr_ptr[i_stride + j], unit) 
                    for j in range(size_j)]
                ) + ")"
            )
        return ",".join(res_l)

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline str _escape_ndarray_td64(np.ndarray arr):
    """(internal) Escape numpy.ndarray with `timedelta64` dtype to literal `<'str'>`.

    :param arr `<'ndarray[timedelta64]'>`: Numpy array in `timedelta64` dtype to escape.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j, i_stride
        int unit
        np.npy_int64* arr_ptr
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return "()"  # exit
        unit = cyutils.get_arr_nptime_unit(arr)
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        res_l = [_escape_timedelta64_value(arr_ptr[i], unit) for i in range(size_i)]
        return "(" + ",".join(res_l) + ")"

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return "()"  # exit
        size_i = shape[0]
        unit = cyutils.get_arr_nptime_unit(arr)
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            res_l.append(
                "(" + ",".join(
                    [_escape_timedelta64_value(arr_ptr[i_stride + j], unit) 
                    for j in range(size_j)]
                ) + ")"
            )
        return ",".join(res_l)

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline str _escape_ndarray_bytes(np.ndarray arr):
    """(internal) Escape numpy.ndarray with `bytes` dtype to literal `<'str'>`.

    :param arr `<'ndarray[bytes]'>`: Numpy array in `bytes` dtype to escape.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return "()"  # exit
        res_l = [escape_bytes(arr_1d_getitem(arr, i)) for i in range(size_i)]
        return "(" + ",".join(res_l) + ")"

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return "()"  # exit
        size_i = shape[0]
        res_l = [
            "(" + ",".join([escape_bytes(arr_2d_getitem(arr, i, j)) for j in range(size_j)]) + ")"
            for i in range(size_i)
        ]
        return ",".join(res_l)

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline str _escape_ndarray_unicode(np.ndarray arr):
    """(internal) Escape numpy.ndarray with `unicode` dtype to literal `<'str'>`.

    :param arr `<'ndarray[unicode]'>`: Numpy array in `unicode` dtype to escape.
    :returns `<'str'>`: Escaped ndarray literal.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return "()"  # exit
        res_l = [escape_str(arr_1d_getitem(arr, i)) for i in range(size_i)]
        return "(" + ",".join(res_l) + ")"

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return "()"  # exit
        size_i = shape[0]
        res_l = [
            "(" + ",".join([escape_str(arr_2d_getitem(arr, i, j)) for j in range(size_j)]) + ")"
            for i in range(size_i)
        ]
        return ",".join(res_l)

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline bint _raise_invalid_array_dim(arr: np.ndarray) except -1:
    """Raise `ValueError` for unsupported ndarray dimension."""
    raise ValueError(
        "Cannot escape %d-dimensional 'ndarray[%s]', "
        "only 1 or 2 dimensioin array is supported." % (arr.ndim, arr.dtype)
    )

# . pandas
cdef inline str escape_series(object data):
    """Escape pandas.Series to literal `<'str'>`.
    
    :param data `<'Series'>`: Pandas Series data to escape.
    :returns `<'str'>`: Escaped Series literal.
    """
    try:
        arr: np.ndarray = data.values
    except Exception as err:
        raise TypeError(
            "Cannot escape %s, unable to access its underlying ndarray "
            "through the 'values' property." % type(data)
        ) from err
    return escape_ndarray(arr)

cdef inline str escape_dataframe(object data):
    """Escape pandas.DataFrame to literal `<'str'>`.

    :prarm data `<'DataFrame'>`: Pandas DataFrame data to escape.
    :returns `<'str'>`: Escaped DataFrame literal.
    """
    cdef tuple shape = data.shape
    if tuple_len(shape) != 2:
        raise ValueError("Cannot escape pandas.DataFrame with unsupported shape: %s." % str(shape))
    cdef Py_ssize_t size_i = <object> tuple_getitem(shape, 0)
    if size_i == 0:
        return "()"  # exit
    cdef Py_ssize_t size_j = <object> tuple_getitem(shape, 1)
    if size_j == 0:
        return "()"  # exit

    # Escape DataFrame
    cdef list cols = [escape_ndarray_items(r.values, False) for _, r in data.items()]
    cdef list rows, row
    cdef tuple col
    cdef Py_ssize_t i, j
    rows = []
    for i in range(size_i):
        row = []
        for j in range(size_j):
            col = <tuple> list_getitem(cols, j)
            row.append(<object> tuple_getitem(col, i))
        rows.append("(" + ",".join(row) + ")")
    return ",".join(rows)

# . dispatch
cdef inline str escape_common(object data):
    """Escape common types to literal `<'str'>`."""
    # Get data type
    dtype = type(data)

    # Basic types
    if dtype is str:
        return escape_str(data)
    if dtype is int:
        return escape_int(data)
    if dtype is float:
        return escape_float(data)
    if dtype is bool:
        return escape_bool(data)
    if data is None:
        return escape_none(data)

    # Date & Time types
    if dtype is datetime.datetime:
        return escape_datetime(data)
    if dtype is datetime.date:
        return escape_date(data)
    if dtype is datetime.time:
        return escape_time(data)
    if dtype is datetime.timedelta:
        return escape_timedelta(data)

    # Bytes types
    if dtype is bytes:
        return escape_bytes(data)

    # Numeric types
    if dtype is T_DECIMAL:
        return escape_decimal(data)

    # Sequence types
    if dtype is tuple:
        return escape_tuple(data)
    if dtype is list:
        return escape_list(data)
    if dtype is set:
        return escape_set(data)

    # Mapping types
    if dtype is dict:
        return escape_dict(data)

    # Uncommon types
    return escape_uncommon(data, dtype)

cdef inline str escape_uncommon(object data, dtype: type):
    """Escape uncommon types to literal `<'str'>`."""
    # Basic types
    if dtype is T_NP_STR:
        return escape_str(data)
    if dtype is T_NP_INT64 or dtype is T_NP_INT32 or dtype is T_NP_INT16 or dtype is T_NP_INT8:
        return escape_int(data)
    if dtype is T_NP_UINT64 or dtype is T_NP_UINT32 or dtype is T_NP_UINT16 or dtype is T_NP_UINT8:
        return escape_int(data)
    if dtype is T_NP_FLOAT64 or dtype is T_NP_FLOAT32 or dtype is T_NP_FLOAT16:
        return escape_float64(data)
    if dtype is T_NP_BOOL:
        return escape_bool(data)

    # Date & Time
    if dtype is T_NP_DATETIME64:
        return escape_datetime64(data)
    if dtype is T_NP_TIMEDELTA64:
        return escape_timedelta64(data)
    if dtype is T_PD_TIMESTAMP:
        return escape_datetime(data)
    if dtype is T_PD_TIMEDELTA:
        return escape_timedelta(data)
    if dtype is T_STRUCT_TIME:
        return escape_struct_time(data)

    # Bytes
    if dtype is bytearray:
        return escape_bytearray(data)
    if dtype is memoryview:
        return escape_memoryview(data)
    if dtype is T_NP_BYTES:
        return escape_bytes(data)

    #  NULL
    if dtype is T_PD_NAT:
        return escape_none(data)

    # Sequence types
    if dtype is frozenset:
        return escape_frozenset(data)
    if dtype is range:
        return escape_range(data)
    if dtype is T_DICT_KEYS or dtype is T_DICT_VALUES:
        return escape_sequence(data)

    # Mapping types
    if dtype is T_DICT_ITEMS:
        return escape_dict(dict(data))

    # Numpy
    if dtype is np.ndarray:
        return escape_ndarray(data)
    if dtype is T_NP_RECORD:
        return escape_sequence(data)

    # Pandas
    if dtype is T_PD_SERIES or dtype is T_PD_DATETIMEINDEX or dtype is T_PD_TIMEDELTAINDEX:
        return escape_series(data)
    if dtype is T_PD_DATAFRAME:
        return escape_dataframe(data)

    # Cytimes
    if dtype is T_PYDT:
        return escape_datetime(data)
    if dtype is T_PDDT:
        return escape_series(data)

    # Subclass
    return escape_subclass(data, dtype)

cdef inline str escape_subclass(object data, dtype: type):
    """Escape subclass types to literal `<'str'>`."""
    # Custom subclass
    if issubclass(dtype, ObjStr):
        return str(data)
    if issubclass(dtype, RawText):
        return _escape_rawtext(data)
    if issubclass(dtype, SQLFunction):
        return _escape_sqlfunction(data)
    if issubclass(dtype, SQLInterval):
        return _escape_sqlinterval(data)

    # Basic subclasses
    if issubclass(dtype, str):
        return escape_str(str(data))
    if issubclass(dtype, int):
        return escape_int(int(data))
    if issubclass(dtype, float):
        return escape_float(float(data))
    if issubclass(dtype, bool):
        return escape_bool(bool(data))

    # Date & Time subclasses
    if issubclass(dtype, datetime.datetime):
        return escape_datetime(data)
    if issubclass(dtype, datetime.date):
        return escape_date(data)
    if issubclass(dtype, datetime.time):
        return escape_time(data)
    if issubclass(dtype, datetime.timedelta):
        return escape_timedelta(data)

    # Bytes subclasses
    if issubclass(dtype, bytes):
        return escape_bytes(bytes(data))
    if issubclass(dtype, bytearray):
        return escape_bytearray(bytearray(data))

    # Sequence subclasses
    if issubclass(dtype, list):
        return escape_list(list(data))
    if issubclass(dtype, tuple):
        return escape_tuple(tuple(data))
    if issubclass(dtype, set):
        return escape_set(set(data))
    if issubclass(dtype, frozenset):
        return escape_frozenset(frozenset(data))

    # Mapping subclasses
    if issubclass(dtype, dict):
        return escape_dict(dict(data))

    # Pandas
    if issubclass(dtype, T_PD_INDEX):
        return escape_series(data)
    if issubclass(dtype, T_PD_SERIES):
        return escape_series(data)

    # Unsupported data type
    raise TypeError("Cannot escape %s, data type is not supported." % dtype)

# Items -----------------------------------------------------------------------------------------------
# . sequence
cdef inline object escape_list_items(list data, bint many):
    """Escape the elements of a list to sequences of literals `<'tuple/list'>`.
    
    :param data `<'list'>`: The list data to escape.
    :param many `<'bool'>`: Specifies how nested sequences (element) is handled.
    
        - If False, escape each element of the list to a single 
          literal string and returns a `tuple`.
        - Otherwise, when the encountered nested element, escape
          it to a `tuple` of literal strings, and returns a `list`.

    :returns `<'tuple/list'>`: Escaped sequences of literals (see examples).

    ## Examples
    ```python
    # many=False & flat
    escape_list_items(["val1", 1, 1.1], False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    # many=True & flat
    escape_list_items(["val1", 1, 1.1], True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    # many=False & nested
    escape_list_items([["val1", 1, 1.1], ["val2", 2, 2.2]], False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    # many=True & nested
    escape_list_items([["val1", 1, 1.1], ["val2", 2, 2.2]], False)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    ```
    """
    if many:
        return [escape_common_items(i, False) for i in data]
    else:
        return tuple([escape_common(i) for i in data])

cdef inline object escape_tuple_items(tuple data, bint many):
    """Escape the elements of a tuple to sequences of literals `<'tuple/list'>`.
    
    :param data `<'tuple'>`: The tuple data to escape.
    :param many `<'bool'>`: Specifies how nested sequences (element) is handled.
    
        - If False, escape each element of the tuple to a single
          literal string and returns a `tuple`.
        - Otherwise, when the encountering nested element, escape
          it to a `tuple` of literal strings, and returns a `list`.

    :returns `<'tuple/list'>`: Escaped sequences of literals (see example).

    ## Examples
    ```python
    # many=False & flat
    escape_tuple_items(("val1", 1, 1.1), False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    # many=True & flat
    escape_tuple_items(("val1", 1, 1.1), True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    # many=False & nested
    escape_tuple_items((["val1", 1, 1.1], ["val2", 2, 2.2]), False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    # many=True & nested
    escape_tuple_items((["val1", 1, 1.1], ["val2", 2, 2.2]), False)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    ```
    """
    if many:
        return [escape_common_items(i, False) for i in data]
    else:
        return tuple([escape_common(i) for i in data])

cdef inline object escape_set_items(set data, bint many):
    """Escape the elements of a set to sequences of literals `<'tuple/list'>`.
    
    :param data `<'set'>`: The set data to escape.
    :param many `<'bool'>`: Specifies how nested sequences (element) is handled.
    
        - If False, escape each element of the set to a single
          literal string and returns a `tuple`.
        - Otherwise, when the encountering nested element, escape
          it to a `tuple` of literal strings, and returns a `list`.

    :returns `<'tuple/list'>`: Escaped sequences of literals (see example).

    ## Examples
    ```python
    # many=False & flat
    escape_set_items({"val1", 1, 1.1}, False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    # many=True & flat
    escape_set_items({"val1", 1, 1.1}, True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    # many=False & nested
    escape_set_items({("val1", 1, 1.1), ("val2", 2, 2.2)}, False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    # many=True & nested
    escape_set_items({("val1", 1, 1.1), ("val2", 2, 2.2)}, False)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    ```
    """
    if many:
        return [escape_common_items(i, False) for i in data]
    else:
        return tuple([escape_common(i) for i in data])

cdef inline object escape_frozenset_items(frozenset data, bint many):
    """Escape the elements of a frozenset to sequences of literals `<'tuple/list'>`.
    
    :param data `<'frozenset'>`: The frozenset data to escape.
    :param many `<'bool'>`: Specifies how nested sequences (element) is handled.
    
        - If False, escape each element of the frozenset to a single
          literal string and returns a `tuple`.
        - Otherwise, when the encountering nested element, escape
          it to a `tuple` of literal strings, and returns a `list`.

    :returns `<'tuple/list'>`: Escaped sequences of literals (see example).

    ## Examples
    ```python
    # many=False & flat
    escape_frozenset_items(frozenset({"val1", 1, 1.1}), False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    # many=True & flat
    escape_frozenset_items(frozenset({"val1", 1, 1.1}), True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    # many=False & nested
    escape_frozenset_items(frozenset({("val1", 1, 1.1), ("val2", 2, 2.2)}), False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    # many=True & nested
    escape_frozenset_items(frozenset({("val1", 1, 1.1), ("val2", 2, 2.2)}), False)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    ```
    """
    if many:
        return [escape_common_items(i, False) for i in data]
    else:
        return tuple([escape_common(i) for i in data])

cdef inline object escape_range_items(object data, bint many):
    """Escape the element of a `range` iterator to sequences of literals `<'tuple/list'>`.

    :param data `<'range'>`: The range iterator to escape.
    :param many `<'bool'>`: If False, returns `tuple` of literal strings;
        otherwise, returns `list` of literal strings of the range values.
    :returns `<'tuple/list'>`: Escaped sequences of literals (see examples).

    ## Examples:
    ```python
    # many=False
    escape_range_items(range(3), False)
    >>> ("0", "1", "2")  # tuple[str]

    # many=True
    escape_range_items(range(3), True)
    >>> ["0", "1", "2"]  # list[str]
    ```
    """
    cdef res = [escape_int(i) for i in data]
    return res if many else tuple(res)

cdef inline object escape_sequence_items(object data, bint many):
    """Escape the elements of a iterable sequence-like object to sequences of literals `<'tuple/list'>`.
    
    :param data `<'sequence'>`: The sequence-like data to escape.
    :param many `<'bool'>`: Specifies how nested sequences (element) is handled.
    
        - If False, escape each element of the sequence to a single
          literal string and returns a `tuple`.
        - Otherwise, when the encountering nested element, escape
          it to a `tuple` of literal strings, and returns a `list`.

    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    if many:
        return [escape_common_items(i, False) for i in data]
    else:
        return tuple([escape_common(i) for i in data])

# . mapping
cdef inline object escape_dict_items(dict data, bint many):
    """Escape the values of a dictionary to sequences of literals `<'tuple/list'>`.
    
    :param data `<'dict'>`: The dictionary data to escape.
    :param many `<'bool'>`: Specifies how nested value is handled.
    
        - If False, escape each value of the dictionary to a single 
          literal string and returns a `tuple`.
        - Otherwise, when the encountered nested value, escape
          it to a `tuple` of literal strings, and returns a `list`.

    :returns `<'tuple/list'>`: Escaped sequences of literals (see examples).

    ## Examples
    ```python
    # many=False & flat
    escape_dict_items({"key1": "val1", "key2": 1, "key3": 1.1}.values(), False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    # many=True & flat
    escape_dict_items({"key1": "val1", "key2": 1, "key3": 1.1}.values(), True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    # many=False & nested
    escape_dict_items({"key1": ["val1", 1, 1.1], "key2": ["val2", 2, 2.2]}.values(), False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    # many=True & nested
    escape_dict_items({"key1": ["val1", 1, 1.1], "key2": ["val2", 2, 2.2]}.values(), True)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    ```
    """
    if many:
        return [escape_common_items(v, False) for v in data.values()]
    else:
        return tuple([escape_common(v) for v in data.values()])

# . numpy
cdef inline object escape_ndarray_items(np.ndarray data, bint many):
    """Escape the elements of a numpy.ndarray to sequences of literals `<'tuple/list'>`.
    
    :param data `<'ndarray'>`: The ndarray data to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    
        - If False, escape each value of the array to a single 
          literal string and returns a `tuple`.
        - Otherwise, when the encountered nested value, escape
          it to a `tuple` of literal strings, and returns a `list`.
        - For 2-dimension array, always escape to `list`.

    :returns `<'tuple/list'>`: Escaped sequences of literals (see examples).

    ## Example
    ```python
    1-dimension & many=False
    escape_ndarray_items(np.array([1, 2, 3]), False)
    >>> ("1", "2", "3")  # tuple[str]

    1-dimension & many=True
    escape_ndarray_items(np.array([1, 2, 3]), True)
    >>> ["1", "2", "3"]  # list[str] or list[tuple[str]]

    2-dimension & many=False is ignored
    escape_ndarray_items(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]), False)
    >>> [("1.1", "2.2", "3.3"), ("4.4", "5.5", "6.6")]  # list[tuple[str]]
    ```
    """
    # Get ndarray dtype
    cdef int dtype = np.PyArray_TYPE(data)

    # Escape
    # . ndarray[object]
    if dtype == np.NPY_TYPES.NPY_OBJECT:
        return _escape_ndarray_object_items(data, many)
    # . ndarray[int] & ndarray[uint]
    if dtype in (
        np.NPY_TYPES.NPY_INT64, np.NPY_TYPES.NPY_INT32,
        np.NPY_TYPES.NPY_INT16, np.NPY_TYPES.NPY_INT8,
        np.NPY_TYPES.NPY_UINT64, np.NPY_TYPES.NPY_UINT32,
        np.NPY_TYPES.NPY_UINT16, np.NPY_TYPES.NPY_UINT8,
    ):
        return _escape_ndarray_int_items(data, many)
    # . ndarray[float]
    if dtype in (
        np.NPY_TYPES.NPY_FLOAT64,
        np.NPY_TYPES.NPY_FLOAT32,
        np.NPY_TYPES.NPY_FLOAT16,
    ):
        return _escape_ndarray_float_items(data, many)
    # . ndarray[bool]
    if dtype == np.NPY_TYPES.NPY_BOOL:
        return _escape_ndarray_bool_items(data, many)
    # . ndarray[datetime64]
    if dtype == np.NPY_TYPES.NPY_DATETIME:
        return _escape_ndarray_dt64_items(data, many)
    # . ndarray[timedelta64]
    if dtype == np.NPY_TYPES.NPY_TIMEDELTA:
        return _escape_ndarray_td64_items(data, many)
    # . ndarray[bytes]
    if dtype == np.NPY_TYPES.NPY_STRING:
        return _escape_ndarray_bytes_items(data, many)
    # . ndarray[unicode]
    if dtype == np.NPY_TYPES.NPY_UNICODE:
        return _escape_ndarray_unicode_items(data, many)

    # Unsupported dtype
    raise TypeError("Cannot escape 'ndarray[%s]', array dtype is not supported." % data.dtype)

cdef inline object _escape_ndarray_object_items(np.ndarray arr, bint many):
    """(internal) Escape the elements of a numpy.ndarray with `object` dtype to sequences of literals `<'tuple/list'>`.
    
    :param arr `<'ndarray[object]'>`: The ndarray in `object` dtype to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return [] if many else ()  # exit
        if many:
            return [escape_common_items(arr_1d_getitem(arr, i), False) for i in range(size_i)]
        else:
            return tuple([escape_common(arr_1d_getitem(arr, i)) for i in range(size_i)])

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return []  # exit
        size_i = shape[0]
        return [
            tuple([escape_common(arr_2d_getitem(arr, i, j)) for j in range(size_j)])
            for i in range(size_i)
        ]

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline object _escape_ndarray_int_items(np.ndarray arr, bint many):
    """(internal) Escape the elements of a numpy.ndarray with `integer` dtype to sequences of literals `<'tuple/list'>`.
    
    :param arr `<'ndarray[int]'>`: The ndarray in `integer` dtype to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        str  res_s
        list res_l

    # 1-dimension
    if ndim == 1:
        if shape[0] == 0:
            return [] if many else ()  # exit
        #: 'res_s' will be like
        #: "[1,2,3]"
        res_s = orjson_dumps_numpy(arr)
        #: remove the outer brackets
        #: "1,2,3"
        res_s = str_substr(res_s, 1, str_len(res_s) - 1)
        #: split to list by comma
        #: ['1', '2', '3']
        res_l = str_split(res_s, ",", -1)
        return res_l if many else tuple(res_l)

    # 2-dimension
    if ndim == 2:
        if shape[1] == 0:
            return []  # exit
        #: 'res_s' will be like
        #: "[[1,2,3],[4,5,6]]"
        res_s = orjson_dumps_numpy(arr)
        #: remove the outer 2 brackets
        #: "1,2,3],[4,5,6"
        res_s = str_substr(res_s, 2, str_len(res_s) - 2)
        #: split to list by "],["
        #: ['1,2,3', '4,5,6']
        res_l = str_split(res_s, "],[", -1)
        #: for each item, split by comma and convert to tuple
        #: [('1', '2', '3'), ('4', '5', '6')]
        return [tuple(str_split(i, ",", -1)) for i in res_l]

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline object _escape_ndarray_float_items(np.ndarray arr, bint many):
    """(internal) Escape the elements of a numpy.ndarray with `float` dtype to sequences of literals `<'tuple/list'>`.
    
    :param arr `<'ndarray[float]'>`: The ndarray in `float` dtype to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j
        str  res_s
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return [] if many else ()  # exit
        # . slow approach for nan & inf
        if not is_arr_1d_float_finite(arr, size_i):
            res_l = _escape_ndarray_1d_float_items_slow(arr, size_i)
            return res_l if many else tuple(res_l)
        #: 'res_s' will be like
        #: "[1.0,2.0,3.0]"
        res_s = orjson_dumps_numpy(arr)
        #: remove the outer brackets
        #: "1.0,2.0,3.0"
        res_s = str_substr(res_s, 1, str_len(res_s) - 1)
        #: split to list by comma
        #: ['1.0', '2.0', '3.0']
        res_l = str_split(res_s, ",", -1)
        return res_l if many else tuple(res_l)

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return []  # exit
        size_i = shape[0]
        # . slow approach for nan & inf
        if not is_arr_2d_float_finite(arr, size_i, size_j):
            return _escape_ndarray_2d_float_items_slow(arr, size_i, size_j)
        #: 'res_s' will be like
        #: "[[1.0,2.0,3.0],[4.0,5.0,6.0]]"
        res_s = orjson_dumps_numpy(arr)
        #: remove the outer 2 brackets
        #: "1.0,2.0,3.0],[4.0,5.0,6.0"
        res_s = str_substr(res_s, 2, str_len(res_s) - 2)
        #: split to list by "],["
        #: ['1.0,2.0,3.0', '4.0,5.0,6.0']
        res_l = str_split(res_s, "],[", -1)
        #: for each item, split by comma and convert to tuple
        #: [('1.0', '2.0', '3.0'), ('4.0', '5.0', '6.0')]
        return [tuple(str_split(i, ",", -1)) for i in res_l]

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline list _escape_ndarray_1d_float_items_slow(np.ndarray arr, np.npy_intp size=-1):
    """(internal) Escape 1-D numpy.ndarray with `float` dtype to sequences 
    of literal through native str (slow) approach `<'list'>`.

    :param arr `<ndarray'>`: The 1-dimensional float array to escape.
    :param size `<'int'>`: Optional size of the ndarray. Defaults to `-1`.
    :returns `<'list'>`: Escaped sequences of literals.
    """
    # Setup
    cdef:
        int dtype = np.PyArray_TYPE(arr)
        np.npy_float64* f64_ptr
        np.npy_float64  f64_v
        np.npy_float32* f32_ptr
        np.npy_float32  f32_v
        np.npy_intp     i
        list res_l
    if size < 0:
        size = arr.shape[0]

    # Escape: float64
    if dtype == np.NPY_TYPES.NPY_FLOAT64:
        f64_ptr = <np.npy_float64*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size):
            f64_v = f64_ptr[i]
            if not math.isfinite(f64_v):
                res_l.append("NULL")
            else:
                res_l.append(str(f64_v))
        return res_l

    # Cast: float16 -> float32
    if dtype == np.NPY_TYPES.NPY_FLOAT16:
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        dtype = np.NPY_TYPES.NPY_FLOAT32

    # Escape: float32
    if dtype == np.NPY_TYPES.NPY_FLOAT32:
        f32_ptr = <np.npy_float32*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size):
            f32_v = f32_ptr[i]
            if not math.isfinite(f32_v):
                res_l.append("NULL")
            else:
                res_l.append(str(f32_v))
        return res_l

    # Unsupported dtype
    raise AssertionError(
        "_escape_ndarray_1d_float_items_slow: The ndarray dtype must be 'float16', 'float32' or 'float64', "
        "instead got 'ndarray[%s]'." % arr.dtype
    )

cdef inline list _escape_ndarray_2d_float_items_slow(np.ndarray arr, np.npy_intp size_i=-1, np.npy_intp size_j=-1):
    """(internal) Escape 2-D numpy.ndarray with `float` dtype to sequences 
    of literal through native str (slow) approach `<'list[tuple]'>`.
    
    :param arr `<ndarray'>`: The 2-dimensional float array to escape.
    :param size_i `<'int'>`: Optional size of the first dimension. Defaults to `-1`.
    :param size_j `<'int'>`: Optional size of the second dimension. Defaults to `-1`.
    :returns `<'list[tuple]'>`: Escaped sequences of literals.
    """
    # Setup
    cdef:
        int dtype = np.PyArray_TYPE(arr)
        np.npy_float64* f64_ptr
        np.npy_float64  f64_v
        np.npy_float32* f32_ptr
        np.npy_float32  f32_v
        np.npy_intp     i, j, i_stride
        list res_l, row_l
    if size_i < 0:
        size_i = arr.shape[0]
    if size_j < 0:
        size_j = arr.shape[1]

    # Escape: float64
    if dtype == np.NPY_TYPES.NPY_FLOAT64:
        f64_ptr = <np.npy_float64*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            row_l = []
            for j in range(size_j):
                f64_v = f64_ptr[i_stride + j]
                if not math.isfinite(f64_v):
                    row_l.append("NULL")
                else:
                    row_l.append(str(f64_v))
            res_l.append(tuple(row_l))
        return res_l

    # Cast: float16 -> float32
    if dtype == np.NPY_TYPES.NPY_FLOAT16:
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        dtype = np.NPY_TYPES.NPY_FLOAT32

    # Escape: float32
    if dtype == np.NPY_TYPES.NPY_FLOAT32:
        f32_ptr = <np.npy_float32*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            row_l = []
            for j in range(size_j):
                f32_v = f32_ptr[i_stride + j]
                if not math.isfinite(f32_v):
                    row_l.append("NULL")
                else:
                    row_l.append(str(f32_v))
            res_l.append(tuple(row_l))
        return res_l

    # Unsupported dtype
    raise AssertionError(
        "_escape_ndarray_2d_float_items_slow: The ndarray dtype must be 'float16', 'float32' or 'float64', "
        "instead got 'ndarray[%s]'." % arr.dtype
    )

cdef inline object _escape_ndarray_bool_items(np.ndarray arr, bint many):
    """(internal) Escape the elements of a numpy.ndarray with `bool` dtype to sequences of literals `<'tuple/list'>`.
    
    :param arr `<'ndarray[bool]'>`: The ndarray in `bool` dtype to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j, i_stride
        np.npy_bool* arr_ptr
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return [] if many else ()  # exit
        arr_ptr = <np.npy_bool*> np.PyArray_DATA(arr)
        res_l = ["1" if arr_ptr[i] else "0" for i in range(size_i)]
        return res_l if many else tuple(res_l)

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return []  # exit
        size_i = shape[0]
        arr_ptr = <np.npy_bool*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            res_l.append(
                tuple(["1" if arr_ptr[i_stride + j] else "0" 
                for j in range(size_j)])
            )
        return res_l

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline object _escape_ndarray_dt64_items(np.ndarray arr, bint many):
    """(internal) Escape the elements of a numpy.ndarray with `datetime64` dtype to sequences of literals `<'tuple/list'>`.
    
    :param arr `<'ndarray[datetime64]'>`: The ndarray in `datetime64` dtype to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j, i_stride
        int unit
        np.npy_int64* arr_ptr
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return [] if many else ()  # exit
        unit = cyutils.get_arr_nptime_unit(arr)
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        res_l = [_escape_datetime64_value(arr_ptr[i], unit) for i in range(size_i)]
        return res_l if many else tuple(res_l)

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return []  # exit
        size_i = shape[0]
        unit = cyutils.get_arr_nptime_unit(arr)
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            res_l.append(
                tuple([_escape_datetime64_value(arr_ptr[i_stride + j], unit) 
                for j in range(size_j)])
            )
        return res_l

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline object _escape_ndarray_td64_items(np.ndarray arr, bint many):
    """(internal) Escape the elements of a numpy.ndarray with `timedelta64` dtype to sequences of literals `<'tuple/list'>`.
    
    :param arr `<'ndarray[timedelta64]'>`: The ndarray in `timedelta64` dtype to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j, i_stride
        int unit
        np.npy_int64* arr_ptr
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return [] if many else ()  # exit
        unit = cyutils.get_arr_nptime_unit(arr)
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        res_l = [_escape_timedelta64_value(arr_ptr[i], unit) for i in range(size_i)]
        return res_l if many else tuple(res_l)

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return []  # exit
        size_i = shape[0]
        unit = cyutils.get_arr_nptime_unit(arr)
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        res_l = []
        for i in range(size_i):
            i_stride = i * size_j
            res_l.append(
                tuple([_escape_timedelta64_value(arr_ptr[i_stride + j], unit) 
                for j in range(size_j)])
            )
        return res_l

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline object _escape_ndarray_bytes_items(np.ndarray arr, bint many):
    """(internal) Escape the elements of a numpy.ndarray with `bytes` dtype to sequences of literals `<'tuple/list'>`.
    
    :param arr `<'ndarray[bytes]'>`: The ndarray in `bytes` dtype to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return [] if many else ()  # exit
        res_l = [escape_bytes(arr_1d_getitem(arr, i)) for i in range(size_i)]
        return res_l if many else tuple(res_l)

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return []  # exit
        size_i = shape[0]
        return [
            tuple([escape_bytes(arr_2d_getitem(arr, i, j)) for j in range(size_j)])
            for i in range(size_i)
        ]

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

cdef inline object _escape_ndarray_unicode_items(np.ndarray arr, bint many):
    """(internal) Escape the elements of a numpy.ndarray with `unicode` dtype to sequences of literals `<'tuple/list'>`.

    :param arr `<'ndarray[unicode]'>`: The ndarray in `unicode` dtype to escape.
    :param many `<'bool'>`: Specifies how nested value is handled. Only applicable to 1-dimensional array.
    :returns `<'tuple/list'>`: Escaped sequences of literals.
    """
    cdef: 
        int ndim = arr.ndim
        np.npy_intp* shape = arr.shape
        np.npy_intp size_i, size_j, i, j
        list res_l

    # 1-dimension
    if ndim == 1:
        size_i = shape[0]
        if size_i == 0:
            return [] if many else ()  # exit
        res_l = [escape_str(arr_1d_getitem(arr, i)) for i in range(size_i)]
        return res_l if many else tuple(res_l)

    # 2-dimension
    if ndim == 2:
        size_j = shape[1]
        if size_j == 0:
            return []  # exit
        size_i = shape[0]
        return [
            tuple([escape_str(arr_2d_getitem(arr, i, j)) for j in range(size_j)])
            for i in range(size_i)
        ]

    # Unsupported dimension
    _raise_invalid_array_dim(arr)

# . pandas
cdef inline object escape_series_items(object data, bint many):
    """Escape the elements of a pandas.Series to sequences of literals `<'tuple/list'>`.
    
    :param data `<'Series'>`: The Series data to escape.
    :param many `<'bool'>`: Specifies how nested value is handled.
    
        - If False, escape each value of the Series to a single 
          literal string and returns a `tuple`.
        - Otherwise, when the encountered nested value, escape
          it to a `tuple` of literal strings, and returns a `list`.

    :returns `<'tuple/list'>`: Escaped sequences of literals 
        (see examples in `escape_ndarray_items()`).
    """
    try:
        arr: np.ndarray = data.values
    except Exception as err:
        raise TypeError(
            "Cannot escape %s, unable to access its underlying ndarray "
            "through the 'values' property." % type(data)
        ) from err
    return escape_ndarray_items(arr, many)

cdef inline list escape_dataframe_items(object data):
    """Escape the rows of elements of a pandas.DataFrame to `list` of tuples of literals `<'list[tuple]'>`.

    :param data `<'DataFrame'>`: The DataFrame data to escape.
    :returns `<'list[tuple]'>`: Escaped rows of sequences of literals.
    """
    cdef tuple shape = data.shape
    if tuple_len(shape) != 2:
        raise ValueError("Cannot escape pandas.DataFrame with unsupported shape: %s." % str(shape))
    cdef Py_ssize_t size_i = <object> tuple_getitem(shape, 0)
    if size_i == 0:
        return []  # exit
    cdef Py_ssize_t size_j = <object> tuple_getitem(shape, 1)
    if size_j == 0:
        return []  # exit

    # Escape DataFrame
    cdef list cols = [escape_ndarray_items(r.values, False) for _, r in data.items()]
    cdef list rows, row
    cdef tuple col
    cdef Py_ssize_t i, j
    rows = []
    for i in range(size_i):
        row = []
        for j in range(size_j):
            col = <tuple> list_getitem(cols, j)
            row.append(<object> tuple_getitem(col, i))
        rows.append(tuple(row))
    return rows

# . dispatch
cdef inline object escape_common_items(object data, bint many):
    """Escape the elements of a common type to sequences of literals `<'str/tuple/list'>`."""
    # Get data type
    dtype = type(data)

    # Basic types
    if dtype is str:
        return escape_str(data)
    if dtype is int:
        return escape_int(data)
    if dtype is float:
        return escape_float(data)
    if dtype is bool:
        return escape_bool(data)
    if data is None:
        return escape_none(data)

    # Date & Time types
    if dtype is datetime.datetime:
        return escape_datetime(data)
    if dtype is datetime.date:
        return escape_date(data)
    if dtype is datetime.time:
        return escape_time(data)
    if dtype is datetime.timedelta:
        return escape_timedelta(data)

    # Bytes types
    if dtype is bytes:
        return escape_bytes(data)

    # Numeric types
    if dtype is T_DECIMAL:
        return escape_decimal(data)

    # Sequence types
    if dtype is tuple:
        return escape_tuple_items(data, many)
    if dtype is list:
        return escape_list_items(data, many)
    if dtype is set:
        return escape_set_items(data, many)

    # Mapping types
    if dtype is dict:
        return escape_dict_items(data, many)

    # Uncommon types
    return escape_uncommon_items(data, many, dtype)

cdef inline object escape_uncommon_items(object data, bint many, type dtype):
    """Escape the elements of a uncommon type to sequences of literals `<'str/tuple/list'>`."""
    # Basic types
    if dtype is T_NP_STR:
        return escape_str(data)
    if dtype is T_NP_INT64 or dtype is T_NP_INT32 or dtype is T_NP_INT16 or dtype is T_NP_INT8:
        return escape_int(data)
    if dtype is T_NP_UINT64 or dtype is T_NP_UINT32 or dtype is T_NP_UINT16 or dtype is T_NP_UINT8:
        return escape_int(data)
    if dtype is T_NP_FLOAT64 or dtype is T_NP_FLOAT32 or dtype is T_NP_FLOAT16:
        return escape_float64(data)
    if dtype is T_NP_BOOL:
        return escape_bool(data)

    # Date & Time
    if dtype is T_NP_DATETIME64:
        return escape_datetime64(data)
    if dtype is T_NP_TIMEDELTA64:
        return escape_timedelta64(data)
    if dtype is T_PD_TIMESTAMP:
        return escape_datetime(data)
    if dtype is T_PD_TIMEDELTA:
        return escape_timedelta(data)
    if dtype is T_STRUCT_TIME:
        return escape_struct_time(data)

    # Bytes
    if dtype is bytearray:
        return escape_bytearray(data)
    if dtype is memoryview:
        return escape_memoryview(data)
    if dtype is T_NP_BYTES:
        return escape_bytes(data)

    #  NULL
    if dtype is T_PD_NAT:
        return escape_none(data)

    # Sequence types
    if dtype is frozenset:
        return escape_frozenset_items(data, many)
    if dtype is range:
        return escape_range_items(data, many)
    if dtype is T_DICT_KEYS or dtype is T_DICT_VALUES:
        return escape_sequence_items(data, many)

    # Mapping types
    if dtype is T_DICT_ITEMS:
        return escape_dict_items(dict(data), many)

    # Numpy
    if dtype is np.ndarray:
        return escape_ndarray_items(data, many)
    if dtype is T_NP_RECORD:
        return escape_sequence_items(data, many)

    # Pandas
    if dtype is T_PD_SERIES or dtype is T_PD_DATETIMEINDEX or dtype is T_PD_TIMEDELTAINDEX:
        return escape_series_items(data, many)
    if dtype is T_PD_DATAFRAME:
        return escape_dataframe_items(data)

    # Cytimes
    if dtype is T_PYDT:
        return escape_datetime(data)
    if dtype is T_PDDT:
        return escape_series_items(data, many)

    # Subclass
    return escape_subclass_items(data, many, dtype)

cdef inline object escape_subclass_items(object data, bint many, type dtype):
    """Escape the elements of a subclass type to sequences of literals `<'str/tuple/list'>`."""
    # Custom subclass
    if issubclass(dtype, ObjStr):
        return str(data)
    if issubclass(dtype, RawText):
        return _escape_rawtext(data)
    if issubclass(dtype, SQLFunction):
        return _escape_sqlfunction(data)
    if issubclass(dtype, SQLInterval):
        return _escape_sqlinterval(data)

    # Basic subclasses
    if issubclass(dtype, str):
        return escape_str(str(data))
    if issubclass(dtype, int):
        return escape_int(int(data))
    if issubclass(dtype, float):
        return escape_float(float(data))
    if issubclass(dtype, bool):
        return escape_bool(bool(data))

    # Date & Time subclasses
    if issubclass(dtype, datetime.datetime):
        return escape_datetime(data)
    if issubclass(dtype, datetime.date):
        return escape_date(data)
    if issubclass(dtype, datetime.time):
        return escape_time(data)
    if issubclass(dtype, datetime.timedelta):
        return escape_timedelta(data)

    # Bytes subclasses
    if issubclass(dtype, bytes):
        return escape_bytes(bytes(data))
    if issubclass(dtype, bytearray):
        return escape_bytearray(bytearray(data))

    # Sequence subclasses
    if issubclass(dtype, list):
        return escape_list_items(list(data), many)
    if issubclass(dtype, tuple):
        return escape_tuple_items(tuple(data), many)
    if issubclass(dtype, set):
        return escape_set_items(set(data), many)
    if issubclass(dtype, frozenset):
        return escape_frozenset_items(frozenset(data), many)

    # Mapping subclasses
    if issubclass(dtype, dict):
        return escape_dict_items(dict(data), many)

    # Pandas
    if issubclass(dtype, T_PD_INDEX):
        return escape_series_items(data, many)
    if issubclass(dtype, T_PD_SERIES):
        return escape_series_items(data, many)

    # Unsupported data type
    raise TypeError("Cannot escape %s, data type is not supported." % dtype)

# Escape Function -------------------------------------------------------------------------------------
cpdef object escape(object data, bint many=False, bint itemize=True):
    """Escape data for SQL binding `<'str/tuple/list[str/tuple]'>`.

    :param data `<'Any'>`: The data to escape, supports:

        - **Python built-ins**:
            int, float, bool, str, None, datetime, date, time,
            timedelta, struct_time, bytes, bytearray, memoryview,
            Decimal, dict, list, tuple, set, frozenset, range
        - **Library [numpy](https://github.com/numpy/numpy)**:
            np.int, np.uint, np.float, np.bool, np.bytes,
            np.str, np.datetime64, np.timedelta64, np.ndarray
        - **Library [pandas](https://github.com/pandas-dev/pandas)**:
            pd.Timestamp, pd.Timedelta, pd.DatetimeIndex,
            pd.TimedeltaIndex, pd.Series, pd.DataFrame
        - **Library [cytimes](https://github.com/AresJef/cyTimes)**:
            cytimes.Pydt, cytimes.Pddt

    :param many `<'bool'>`: Whether the `data` is multi-row data. Defaults to `False`.

        - `many=False`: The 'itemize' parameter determines how to escape the `data`.
        - `many=True`: The 'itemize' parameter is ignored, and the `data` type determines how escape is done.
            * 1. Sequence or Mapping (e.g. `list`, `tuple`, `dict`, etc) escapes to `<'list[str]'>`.
            * 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to `<'list[str]'>`.
            * 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes to `<'list[tuple[str]]'>`.
            * 4. Single object (such as `int`, `float`, `str`, etc) escapes to one literal string `<'str'>`.

    :param itemize `<'bool'>`: Whether to escape items of the `data` individually. Defaults to `True`.

        - `itemize=False`: Always escapes to one single literal string `<'str'>`, regardless of the `data` type.
        - `itemize=True`: The data type determines how escape is done.
            * 1. Sequence or Mapping (e.g. `list`, `tuple`, `dict`, etc) escapes to `<'tuple[str]'>`.
            * 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to `<'tuple[str]'>`.
            * 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes to `<'list[tuple[str]]'>`.
            * 4. Single object (such as `int`, `float`, `str`, etc) escapes to one literal string `<'str'>`.

    :returns `<'str/tuple/list'>`:

        - If returns `<'str'>`, it represents a single literal string.
        - If returns `<'tuple'>`, it represents a single row of literal strings.
        - If returns `<'list'>`, it represents multiple rows of literal strings.
    """
    try:
        if itemize or many:
            return escape_common_items(data, many)
        else:
            return escape_common(data)
    except ValueError as err:
        raise errors.EscapeValueError(err) from err
    except TypeError as err:
        raise errors.EscapeTypeError(err) from err
    except AssertionError as err:
        raise err
    except Exception as err:
        raise errors.EscapeError(err) from err

# Decode ==============================================================================================
cdef inline object _decode_string(bytes value, const char* encoding, bint is_binary):
    """(internal) Decode the value from a CHAR/BINARY field `<'str/bytes'>`.

    :param value `<'bytes'>`: The value from the CHAR/BINARY field.
    :param encoding `<'char*'>`: The encoding of the field.
    :param is_binary `<'bool'>`: Indicates whether the field is a BINARY-type column.
        If True, return the `value` as-is (`bytes`) for BINARY field;
        otherwise, deocde by the specifed `encoding` to `<'str'>` for CHAR field.
    :returns `<'str/bytes'>`: The decoded python object.

    ## Example:
    ```python
    # is_binary=True
    _decode_string(b"binary", b"utf8", True)
    >>> b"binary"  # <'bytes'>

    # is_binary=False
    _decode_string(b"char", b"utf8", False)
    >>> "char"     # <'str'>
    """
    # Return as bytes
    if is_binary:
        return value
    # Decode to string
    try:
        return decode_bytes(value, encoding)
    except Exception as err:
        raise errors.DecodeValueError(
            "Cannot decode string from %s.\n"
            "%s" % (value, err)
        ) from err

cdef inline object _decode_integer(bytes value):
    """(internal) Decode the value from an INTEGER field `<'int'>`.

    :param value `<'bytes'>`: The value from the INTEGER field.
    :returns `<'int'>`: The decoded python object.

    ## Example:
    ```python
    # negative
    _decode_int(b'-9223372036854775808')
    >>> -9223372036854775808  # <'int'>

    # positive
    _decode_int(b"18446744073709551615")
    >>> 18446744073709551615  # <'int'>
    ```
    """
    cdef: 
        Py_ssize_t  chs_len = bytes_len(value), i
        const char* chs     = bytes_to_chars(value)
        char        ch0     = chs[0]
        char        ch
        unsigned long long res

    # Negative
    if ch0 == 45:  # '-'
        res = 0
        for i in range(1, chs_len):
            ch = chs[i]
            if not is_ascii_digit(ch):
                raise errors.DecodeValueError("Cannot decode integer from %s." % value)
            res = res * 10 + (chs[i] - 48)
        return -(<long long> res)

    # Positive
    else:
        if not is_ascii_digit(ch0):
            raise errors.DecodeValueError("Cannot decode integer from %s." % value)
        res = ch0 - 48
        for i in range(1, chs_len):
            ch = chs[i]
            if not is_ascii_digit(ch):
                raise errors.DecodeValueError("Cannot decode integer from %s." % value)
            res = res * 10 + (chs[i] - 48)
        return res

cdef inline object _decode_float(bytes value):
    """(internal) Decode the value from a FLOAT field `<'float'>`.

    :param value `<'bytes'>`: The value from the FLOAT field.
    :returns `<'float'>`: The decoded python object.

    ## Example
    ```python
    _decode_float(b'-3.1415')
    >>> -3.1415  # <'float'>
    """
    cdef:
        Py_ssize_t  chs_len = bytes_len(value), i
        const char* chs     = bytes_to_chars(value)
        char        ch0     = chs[0]
        char        ch
        bint seen_dot
        unsigned long long i_part
        unsigned long long f_part = 0
        double f_scale = 1.0

    # Negative
    if ch0 == 45:  # '-'
        i_part = 0; seen_dot = False
        for i in range(1, chs_len):
            ch = chs[i]
            if is_ascii_digit(ch):
                if not seen_dot:
                    i_part = i_part * 10 + (ch - 48)
                else:
                    f_part = f_part * 10 + (ch - 48)
                    f_scale *= 0.1
            elif ch == 46:  # '.'
                seen_dot = True
            else:
                raise errors.DecodeValueError("Cannot decode float from %s." % value)
        return -((<double> i_part) + (<double> f_part) * f_scale)

    # Positive
    else:
        if ch0 == 46:  # '.'
            seen_dot = True;  i_part = 0
        else:
            if not is_ascii_digit(ch0):
                raise errors.DecodeValueError("Cannot decode float from %s." % value)
            seen_dot = False; i_part = ch0 - 48
        for i in range(1, chs_len):
            ch = chs[i]
            if is_ascii_digit(ch):
                if not seen_dot:
                    i_part = i_part * 10 + (ch - 48)
                else:
                    f_part = f_part * 10 + (ch - 48)
                    f_scale *= 0.1
            elif ch == 46:  # '.'
                seen_dot = True
            else:
                raise errors.DecodeValueError("Cannot decode float from %s." % value)
        return ((<double> i_part) + (<double> f_part) * f_scale)

cdef inline object _decode_decimal(bytes value, bint use_decimal):
    """(internal) Decode the value from a DECIMAL field `<'float/Decimal'>`.

    :param value `<'bytes'>`: The value from the DECIMAL field.
    :param use_decimal `<'bool'>`: Specifies how the `value` from a DECIMAL field is decoded.
        If True, decode the value as `<'Decimal'>`; otherwise decode as `<'float'>`.
    :returns `<'float/Decimal'>`: The decoded python object.

    ## Example
    ```python
    # use_decimal=True
    _decode_decimal(b'-3.1415', True)
    >>> Decimal('-3.1415')  # <'Decimal'>

    # use_decimal=False
    _decode_decimal(b'-3.1415', False)
    >>> -3.1415             # <'float'>
    """
    # Decode as float
    if not use_decimal:
        return _decode_float(value)

    # Decode as Decimal
    try:
        return T_DECIMAL(decode_bytes_ascii(value))
    except Exception as err:
        raise errors.DecodeValueError(
            "Cannot decode decimal from %s.\n"
            "%s" % (value, err)
        ) from err

cdef inline object _decode_bit(bytes value, bint decode_bit):
    """(internal) Decode the value from a BIT field `<'bytes/int'>`.

    :param value `<'bytes'>`: The value from the BIT field.
    :param decode_bit `<'bool'>`: Specifies how the `value` from a BIT field is decoded.
        If True, decode the `value` to `<'int'>`; otherwise, return as-is (`bytes`).
    :returns `<'bytes/int'>`: The decoded python object.

    ## Example
    ```python
    # decode_bit=True
    _decode_bit(b"\\x01", True)
    >>> 1         # <'int'>

    # decode_bit=False
    _decode_bit(b"\\x01", False)
    >>> b"\\x01"  # <'bytes'>
    """
    if not decode_bit:
        return value
    return unpack_uint_big_endian(bytes_to_chars(value), bytes_len(value), 0)

cdef inline object _decode_date(bytes value):
    """(internal) Decode the value from a DATE field `<'datetime.date/str'>`

    :param value `<'bytes'>`: The value from a DATE field.
    :returns `<'datetime.date/str'>`: The decoded python object.
        If feild to parsed the Y/M/D components or any date values is out of range,
        return `<'str'>` by decoding the `value` with ascii encoding.

    ## Example
    ```python
    # valid date
    _decode_date(b'2007-02-26')
    >>> datetime.date(2007, 2, 26)  # <'datetime.date'>

    # invalid date
    _decode_date(b'2007-02-30')
    >>> '2007-02-30'                # <'str'>
    ```
    """
    cdef: 
        Py_ssize_t  chs_len = bytes_len(value), i = 0
        const char* chs     = bytes_to_chars(value)
        char        ch      = 0
        int yy, mm, dd, count
    if chs_len < 5:  # min length 'Y-M-D'
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit

    # Year
    yy = count = 0
    while i < chs_len and count < 4:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        yy = yy * 10 + (ch - 48)
        i += 1; count += 1
    if not (1 <= yy <= 9999):
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit
    i += 1  # skip date separator

    # Month
    mm = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        mm = mm * 10 + (ch - 48)
        i += 1; count += 1
    if not (1 <= mm <= 12):
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit
    i += 1  # skip date separator

    # Day
    dd = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        dd = dd * 10 + (ch - 48)
        i += 1; count += 1
    if not (1 <= dd <= 31):
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit

    # Compose date
    try:
        return datetime.date_new(yy, mm, dd)
    except Exception:
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit

cdef inline object _decode_datetime(bytes value):
    """(internal) Decode the value from a DATETIME/TIMESTAMP field `<'datetime.datetime/str'>`.

    :param value `<'bytes'>`: The value from a DATETIME/TIMESTAMP field.
    :returns `<'datetime.datetime/str'>`: The decoded python object.
        If feild to parsed the Y/M/D components or any datetime values is out of range,
        return `<'str'>` by decoding the `value` with ascii encoding.

    ## Example:
    ```
    # valid datetime
    _decode_datetime(b'2007-02-25 23:06:20.123')
    >>> datetime.datetime(2007, 2, 25, 23, 6, 20, 123000)  # <'datetime.datetime'>

    # invalid datetime
    _decode_datetime(b'2007-02-30 23:06:20.123')
    >>> '2007-02-30 23:06:20.123'                          # <'str'>
    ```
    """
    cdef: 
        Py_ssize_t  chs_len = bytes_len(value), i = 0
        const char* chs     = bytes_to_chars(value)
        char        ch      = 0
        int yy, mm, dd, hh, mi, ss, us, count
    if chs_len < 5:  # min length 'Y-M-D'
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit

    # Year
    yy = count = 0
    while i < chs_len and count < 4:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        yy = yy * 10 + (ch - 48)
        i += 1; count += 1
    if not (1 <= yy <= 9999):
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit
    i += 1  # skip date separator

    # Month
    mm = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        mm = mm * 10 + (ch - 48)
        i += 1; count += 1
    if not (1 <= mm <= 12):
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit
    i += 1  # skip date separator

    # Day
    dd = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        dd = dd * 10 + (ch - 48)
        i += 1; count += 1
    if not (1 <= dd <= 31):
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit
    i += 1  # skip date & time separator

    # Hour
    hh = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        hh = hh * 10 + (ch - 48)
        i += 1; count += 1
    i += 1  # skip time separator

    # Minute
    mi = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        mi = mi * 10 + (ch - 48)
        i += 1; count += 1
    i += 1  # skip time separator

    # Second
    ss = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        ss = ss * 10 + (ch - 48)
        i += 1; count += 1
    i += 1  # skip fraction separator

    # Microsecond
    us = count = 0
    while i < chs_len and count < 6:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        us = us * 10 + (ch - 48)
        i += 1; count += 1
    if us > 0:
        while count < 6:
            us *= 10; count += 1

    # Compose datetime
    try:
        return datetime.datetime_new(yy, mm, dd, hh, mi, ss, us, None, 0)
    except Exception:
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit

cdef inline object _decode_time(bytes value):
    """(internal) Decode the value from a TIME field `<'datetime.timedelta/str'>`.

    :param value `<'bytes'>`: The value from a TIME field.
    :returns `<'datetime.timedelta/str'>`: The decoded python object.
        If feild to parsed the `value` as H/M/S/F components, or the values are out of range, 
        return `<'str'>` by decoding the `value` with ascii encoding.

    ## Example
    ```python
    # valid time
    _decode_time(b'25:06:17.123')
    >>> datetime.timedelta(days=1, seconds=3977, microseconds=123000)  # <'datetime.timedelta'>

    # invalid
    _decode_timedelta(b'random crap')
    >>> "random crap"                                                  # <'str'>
    ```
    """
    cdef:
        Py_ssize_t  chs_len = bytes_len(value), i = 0
        const char* chs     = bytes_to_chars(value)
        char        ch      = 0
        long long   hh, mi, ss, us, count
        bint        negate
    if chs_len < 5:  # min length 'H:M:S'
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit

    # Hour
    cdef char ch0 = chs[0]
    if ch0 == 45:  # '-'
        negate = True;  count = 0; hh = 0
    else:
        if not is_ascii_digit(ch0):
            return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit
        negate = False; count = 1; hh = ch0 - 48
    i += 1
    while i < chs_len and count < 6:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        hh = hh * 10 + (ch - 48)
        i += 1; count += 1
    i += 1  # skip time seperator

    # Minute
    mi = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        mi = mi * 10 + (ch - 48)
        i += 1; count += 1
    if not mi <= 59:
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit
    i += 1  # skip time seperator

    # Second
    ss = count = 0
    while i < chs_len and count < 2:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        ss = ss * 10 + (ch - 48)
        i += 1; count += 1
    if not ss <= 59:
        return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit
    i += 1  # skip fraction separator

    # Microsecond
    us = count = 0
    while i < chs_len and count < 6:
        ch = chs[i]
        if not is_ascii_digit(ch):
            break
        us = us * 10 + (ch - 48)
        i += 1; count += 1
    if us > 0:
        while count < 6:
            us *= 10; count += 1

    # Compose: Positive
    cdef long long days, r
    if not negate:
        with cython.cdivision(True):
            days = hh / 24; r = hh % 24
        try:
            return datetime.timedelta_new(days, r * 3600 + mi * 60 + ss, us)
        except Exception:
            return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit

    # Compose: Negative
    else:
        ss += hh * cyutils.SS_HOUR + mi * cyutils.SS_MINUTE
        with cython.cdivision(True):
            days = ss / cyutils.SS_DAY; r = ss % cyutils.SS_DAY
        try:
            return datetime.timedelta_new(-days, -r, -us)
        except Exception:
            return PyUnicode_DecodeASCII(chs, chs_len, b"surrogateescape")  # exit

cdef inline object _decode_json(bytes value, const char* encoding, bint decode_json):
    """(internal) Decode the value from a JSON field `<'Any'>`.

    :param value `<'bytes'>`: The value from a JSON field.
    :param encoding `<'char*'>`: The encoding of the field.
    :param decode_json `<'bool'>`: Specifies how the `value` from a JSON field is decoded.
        If True, deserialized the `value` from JSON string to the corresponding
        python object; otherwise, decode as raw JSON string `<'str'>`.
    :returns `<'str/Any'>`: The decoded python object.

    ## Example
    ```python
    # decode_json=True
    _decode_json(b'{"key": "value", "num": 123}', b"utf8", True)
    >>> {'key': 'value', 'num': 123}    # <'dict'>

    # decode_json=False
    _decode_json(b'{"key": "value", "num": 123}', b"utf8", False)
    >>> '{"key": "value", "num": 123}'  # <'str'>
    ```
    """
    cdef object decoded_str = _decode_string(value, encoding, False)
    if not decode_json:
        return decoded_str
    try:
        return orjson_loads(decoded_str)
    except Exception as err:
        raise errors.DecodeValueError(
            "Cannot decode JSON string '%s'.\n"
            "%s" % (decoded_str, err)
        ) from err

# Decode Function -------------------------------------------------------------------------------------
cpdef object decode(bytes value, unsigned int field_type, const char* encoding, bint is_binary, bint use_decimal=False, bint decode_bit=False, bint decode_json=False):
    """Decode a raw MySQL field value into the appropriate Python object `<'Any'>`.

    This function converts the raw byte sequence returned by the MySQL server
    into a high-level Python representation based on the field type and the
    provided decoding options.

    :param value `<'bytes'>`: The raw byte value received from the server for a single column.
    :param field_type `<'int'>`: The MySQL field type identifier. See `constants.FIELD_TYPE`.
    :param encoding `<'char*/bytes'>`: The character encoding of the column (e.g. `b'utf8'`).
    :param is_binary `<'bool'>`: Indicates whether the field is a BINARY-type column.
    :param use_decimal `<'bool'>`: Specifies how the `value` from a DECIMAL field is decoded. Defaults to `False`.
        If True, decode the value as `<'Decimal'>`; otherwise decode as `<'float'>`.
    :param decode_bit `<'bool'>`: Specifies how the `value` from a BIT field is decoded. Defaults to `False`.
        If True, decode the `value` to `<'int'>`; otherwise, return as-is (`bytes`).
    :param decode_json `<'bool'>`: Specifies how the `value` from a JSON field is decoded. Defaults to `False`.
        If True, deserialized the `value` from JSON string to the corresponding
        python object; otherwise, decode as raw JSON string `<'str'>`.
    :returns `<'Any'>`: The decoded python object.
    """
    # Char / Binary / TEXT / BLOB
    if field_type in (
        _FIELD_TYPE.STRING,         # CHAR / BINARY 254
        _FIELD_TYPE.VAR_STRING,     # VARCHAR / VARBINARY 253
        _FIELD_TYPE.VARCHAR,        # VARCHAR / VARBINARY 15
        _FIELD_TYPE.TINY_BLOB,      # TINYTEXT / TINYBLOB 249
        _FIELD_TYPE.BLOB,           # TEXT / BLOB 252
        _FIELD_TYPE.MEDIUM_BLOB,    # MEDIUMTEXT / MEDIUMBLOB 250
        _FIELD_TYPE.LONG_BLOB,      # LONGTEXT / LONGBLOB 251
    ):
        return _decode_string(value, encoding, is_binary)

    # Integer
    if field_type in (
        _FIELD_TYPE.TINY,           # TINYINT 1
        _FIELD_TYPE.LONGLONG,       # BIGINT 8
        _FIELD_TYPE.LONG,           # INT 3
        _FIELD_TYPE.INT24,          # MEDIUMINT 9
        _FIELD_TYPE.SHORT,          # SMALLINT 2
        _FIELD_TYPE.YEAR,           # YEAR 13
    ):
        return _decode_integer(value)

    # Decimal
    if field_type in (
        _FIELD_TYPE.NEWDECIMAL,     # DECIMAL 246
        _FIELD_TYPE.DECIMAL,        # DECIMAL 0
    ):
        return _decode_decimal(value, use_decimal)

    # Float
    if field_type in (
        _FIELD_TYPE.DOUBLE,         # DOUBLE 5
        _FIELD_TYPE.FLOAT,          # FLOAT 4
    ):
        return _decode_float(value)

    # DATETIME / TIMESTAMP
    if field_type in (
        _FIELD_TYPE.DATETIME,       # DATETIME 12
        _FIELD_TYPE.TIMESTAMP,      # TIMESTAMP 7
    ):
        return _decode_datetime(value)

    # DATE
    if field_type in (
        _FIELD_TYPE.DATE,           # DATE 10
        _FIELD_TYPE.NEWDATE,        # DATE 14
    ):
        return _decode_date(value)

    # TIME
    if field_type in (
        _FIELD_TYPE.TIME,           # TIME 11
    ):
        return _decode_time(value)

    # BIT
    if field_type in (
        _FIELD_TYPE.BIT,            # BIT 16
    ):
        return _decode_bit(value, decode_bit)

    # ENUMERATED
    if field_type in (
        _FIELD_TYPE.ENUM,           # ENUM 247
        _FIELD_TYPE.SET,            # SET 248
    ):
        return _decode_string(value, encoding, False)

    # JSON
    if field_type in (
        _FIELD_TYPE.JSON,           # JSON 245
    ):
        return _decode_json(value, encoding, decode_json)

    # GEOMETRY
    if field_type in (
        _FIELD_TYPE.GEOMETRY,       # GEOMETRY 255
    ):
        return _decode_string(value, encoding, is_binary)

    # Fallback
    try:
        return _decode_string(value, encoding, is_binary)
    except Exception as err:
        raise errors.DecodeTypeError("Cannot decode unsupported field type %d." % field_type) from err
