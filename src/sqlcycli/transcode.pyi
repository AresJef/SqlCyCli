import cython
from typing import Any

# Custom Escape Types -------------------------------------------------------------------------
class ObjStr:
    """For any subclass of <'ObjStr'>, the `escape()` function will
    call its '__str__' method and use the result as the escaped value.

    The '__str__' method must be implemented in the subclass.
    """

    def __str__(self) -> str: ...

class CustomEscapeType(ObjStr):
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

    def __init__(self, obj: Any) -> None:
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

    @property
    def obj(self) -> Any:
        """Returns the underlying object to be escaped `<'Any'>`."""

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class BIT(CustomEscapeType):
    """Represents a value of MySQL BIT column. Act as a wrapper
    for the BIT `obj`, so the `escape()` function can identify
    and escape the value to the desired literal format.

    - Accepts raw bytes or integer value.
    - Validation & conversion only happens when processed by the `escape()` function.
    """

    def __init__(self, obj: Any) -> None:
        """Represents a value of MySQL BIT column. Act as a wrapper
        for the BIT `obj`, so the `escape()` function can identify
        and escape the value to the desired literal format.

        - Accepts raw bytes or integer value.
        - Validation & conversion only happens when processed by the `escape()` function.
        """

class JSON(CustomEscapeType):
    """Represents a value for MySQL JSON column. Act as a wrapper
    for the JSON `obj`, so the `escape()` function can identify and
    escape the value to the desired literal format.

    - Accepts any objects that can be serialized to JSON format.
    - Do `NOT` pass already serialized JSON string to this class.
    - Validation & conversion only happens when processed by the `escape()` function.
    """

    def __init__(self, obj: Any) -> None:
        """Represents a value for MySQL JSON column. Act as a wrapper
        for the JSON `obj`, so the `escape()` function can identify and
        escape the value to the desired literal format.

        - Accepts any objects that can be serialized to JSON format.
        - Do `NOT` pass already serialized JSON string to this class.
        - Validation & conversion only happens when processed by the `escape()` function.
        """

# Escape --------------------------------------------------------------------------------------
def escape(
    data: Any,
    many: bool = False,
    itemize: bool = True,
) -> str | tuple[str] | list[str | tuple[str]]:
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

# Decode --------------------------------------------------------------------------------------
def decode(
    value: bytes,
    field_type: int,
    encoding: cython.p_char | bytes,
    is_binary: bool,
    use_decimal: bool = False,
    decode_bit: bool = False,
    decode_json: bool = False,
) -> Any:
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
