import numpy as np, pandas as pd
import datetime, time, unittest, decimal
from sqlcycli import errors
from sqlcycli.constants import FIELD_TYPE
from sqlcycli.connection import Connection
from sqlcycli.transcode import escape, decode


class TestCase(unittest.TestCase):
    name: str = "Case"
    unix_socket: str = None
    db: str = "test"
    tb: str = "test_table"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "password",
    ) -> None:
        super().__init__("runTest")
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self._start_time = None
        self._ended_time = None

    @property
    def table(self) -> str:
        if self.db is not None and self.tb is not None:
            return f"{self.db}.{self.tb}"
        return None

    def test_all(self) -> None:
        pass

    # utils
    def get_conn(self, **kwargs) -> Connection:
        conn = Connection(
            host=self.host,
            user=self.user,
            password=self.password,
            unix_socket=self.unix_socket,
            local_infile=True,
            **kwargs,
        )
        conn.connect()
        return conn

    def setup(self, table: str = None, **kwargs) -> Connection:
        conn = self.get_conn(**kwargs)
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {self.db};")
            cur.execute(f"DROP TABLE IF EXISTS {self.db}.{tb}")
        return conn

    def drop(self, conn: Connection, table: str = None) -> None:
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute(f"drop table if exists {self.db}.{tb}")

    def delete(self, conn: Connection, table: str = None) -> None:
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute(f"delete from {self.db}.{tb}")

    def log_start(self, msg: str) -> None:
        msg = "START TEST '%s': %s" % (self.name, msg)
        print(msg.ljust(60), end="\r")
        self._start_time = time.perf_counter()

    def log_ended(self, msg: str, skip: bool = False) -> None:
        self._ended_time = time.perf_counter()
        msg = "%s TEST '%s': %s" % ("SKIP" if skip else "PASS", self.name, msg)
        if self._start_time is not None:
            msg += " (%.6fs)" % (self._ended_time - self._start_time)
        print(msg.ljust(60))


class TestEscape(TestCase):
    name: str = "Escape"

    def test_all(self) -> None:
        self.test_escape_bool()
        self.test_escape_int()
        self.test_escape_float()
        self.test_escape_str()
        self.test_escape_none()
        self.test_escape_datetime()
        self.test_escape_date()
        self.test_escape_time()
        self.test_escape_timedelta()
        self.test_escape_bytes()
        self.test_escape_decimal()
        self.test_escape_list_and_tuple()
        self.test_escape_set_and_frozenset()
        self.test_escape_range()
        self.test_escape_sequence()
        self.test_escape_dict()
        self.test_escape_array_object()
        self.test_escape_array_int()
        self.test_escape_array_uint()
        self.test_escape_array_float()
        self.test_escape_array_bool()
        self.test_escape_array_dt64()
        self.test_escape_array_td64()
        self.test_escape_array_bytes()
        self.test_escape_dataframe()
        self.test_escape_custom()

    def test_escape_bool(self) -> None:
        test = "ESCAPE BOOL"
        self.log_start(test)

        test_cases = [(True, "1"), (False, "0")]
        for data, expt in test_cases:
            for dtype in (bool, np.bool_):
                self.assertEqualEscape(expt, dtype(data))

        self.log_ended(test)

    def test_escape_int(self) -> None:
        test = "ESCAPE INT"
        self.log_start(test)

        # signed integer
        for data in range(-10, 11):
            expt = str(data)
            for dtype in (int, np.int8, np.int16, np.int32, np.int64):
                self.assertEqualEscape(expt, dtype(data))

        # unsigned integer
        for data in range(11):
            expt = str(data)
            for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                self.assertEqualEscape(expt, dtype(data))

        self.log_ended(test)

    def test_escape_float(self) -> None:
        test = "ESCAPE FLOAT"
        self.log_start(test)

        for data in (-2.2, -1.1, 0.0, 1.1, 2.2):
            expt = str(data)
            for dtype in (float, np.float16, np.float32, np.float64):
                self.assertEqualEscape(expt, dtype(data))

        with self.assertRaises(errors.EscapeValueError):
            escape(float("nan"))
        with self.assertRaises(errors.EscapeValueError):
            escape(float("inf"))
        with self.assertRaises(errors.EscapeValueError):
            escape(np.nan)
        with self.assertRaises(errors.EscapeValueError):
            escape(np.inf)

        self.log_ended(test)

    def test_escape_str(self) -> None:
        test = "ESCAPE STR"
        self.log_start(test)

        test_cases = [
            # 1. Trivial / baseline
            ("", "''"),
            ("simple", "'simple'"),
            # 2. Quotes
            ("O'Reilly", "'O\\'Reilly'"),  # single quote in middle
            ('He said "hi"', "'He said \\\"hi\\\"'"),
            ("'", "'\\''"),  # bare single quote
            ('"', "'\\\"'"),  # bare double quote
            # 3. Backslash
            ("back\\slash", "'back\\\\slash'"),
            # 4. Newline / carriage return
            ("line1\nline2", "'line1\\nline2'"),
            ("carriage\rreturn", "'carriage\\rreturn'"),
            # 5. NUL and Ctrl-Z
            ("null\0byte", "'null\\0byte'"),
            ("ctrl_z\x1aend", "'ctrl_z\\Zend'"),
            ("\x00", "'\\0'"),  # NUL only
            ("\x1a", "'\\Z'"),  # ASCII 26 only
            # 6. Combined specials
            ("mix '\"\\\n\r\x1a end", "'mix \\'\\\"\\\\\\n\\r\\Z end'"),
            # 7. Non-ASCII (should pass through unchanged except for outer quotes)
            ("你好", "'你好'"),
            ("café", "'café'"),
            ("emoji 😀", "'emoji 😀'"),
            # 8. Other ASCII that should *not* be touched
            ("tab\ttest", "'tab\ttest'"),  # \t is unchanged
            ("space and !@#", "'space and !@#'"),
            ("control\x07bell", "'control\x07bell'"),  # BEL (0x07) unchanged
            # 9. Long string
            ("a" * 1000, "'" + "a" * 1000 + "'"),
            # 10. different languages
            ("中国\n한국어\nにほんご\nEspañol", "'中国\\n한국어\\nにほんご\\nEspañol'"),
        ]
        for data, expt in test_cases:
            for dtype in (str, np.str_):
                self.assertEqualEscape(expt, dtype(data))

        self.log_ended(test)

    def test_escape_none(self) -> None:
        test = "ESCAPE NONE"
        self.log_start(test)

        for many in (True, False):
            for itemize in (True, False):
                for val in (None,):
                    self.assertEqual(escape(val, many, itemize), "NULL")
                    self.assertEqual(escape(val, many, itemize), "NULL")
                    self.assertEqual(escape(val, many, itemize), "NULL")

        self.log_ended(test)

    def test_escape_datetime(self) -> None:
        import pendulum as pl
        from cytimes import Pydt

        test = "ESCAPE DATETIME"
        self.log_start(test)

        test_cases = [
            # fmt: off
            (datetime.datetime(2021, 1, 1, 0, 0, 0), "'2021-01-01 00:00:00'"),
            (datetime.datetime(2021, 1, 1, 0, 0, 0, 123456), "'2021-01-01 00:00:00.123456'"),
            (pl.datetime(2021, 1, 1, 0, 0, 0, tz=None), "'2021-01-01 00:00:00'"),
            (pl.datetime(2021, 1, 1, 0, 0, 0, 123456, tz=None), "'2021-01-01 00:00:00.123456'"),
            # fmt: on
        ]
        for dt, expt in test_cases:
            for data in (
                dt,
                np.datetime64(dt),
                pd.Timestamp(dt),
                Pydt.fromdatetime(dt),
            ):
                self.assertEqualEscape(expt, data)

        data = time.struct_time((2021, 1, 1, 12, 0, 0, 0, 1, 0))
        expt = "'2021-01-01 12:00:00'"
        self.assertEqualEscape(expt, data)

        self.log_ended(test)

    def test_escape_date(self) -> None:
        import pendulum as pl

        test = "ESCAPE DATE"
        self.log_start(test)

        expt = "'2021-01-01'"
        for data in (datetime.date(2021, 1, 1), pl.date(2021, 1, 1)):
            self.assertEqualEscape(expt, data)

        self.log_ended(test)

    def test_escape_time(self) -> None:
        import pendulum as pl

        test = "ESCAPE TIME"
        self.log_start(test)

        for data, expt in (
            (datetime.time(12, 0, 0), "'12:00:00'"),
            (datetime.time(12, 0, 0, 100), "'12:00:00.000100'"),
            (pl.time(12, 0, 0), "'12:00:00'"),
            (pl.time(12, 0, 0, 100), "'12:00:00.000100'"),
        ):
            self.assertEqualEscape(expt, data)

        self.log_ended(test)

    def test_escape_timedelta(self) -> None:
        test = "ESCAPE TIMEDELTA"
        self.log_start(test)

        test_cases = [
            # fmt: off
            (datetime.timedelta(days=1, hours=12, minutes=30, seconds=30), "'36:30:30'"),
            (datetime.timedelta(days=1, hours=12, minutes=30, seconds=30, microseconds=1), "'36:30:30.000001'"),
            (datetime.timedelta(days=-1, hours=12, minutes=30, seconds=30), "'-11:29:30'"),
            (datetime.timedelta(days=-1, hours=12, minutes=30, seconds=30, microseconds=1), "'-11:29:29.999999'"),
            (-datetime.timedelta(days=1, hours=12, minutes=30, seconds=30), "'-36:30:30'"),
            (-datetime.timedelta(days=1, hours=12, minutes=30, seconds=30, microseconds=1), "'-36:30:30.000001'"),
            (-datetime.timedelta(days=-1, hours=12, minutes=30, seconds=30), "'11:29:30'"),
            (-datetime.timedelta(days=-1, hours=12, minutes=30, seconds=30, microseconds=1), "'11:29:29.999999'"),
            # fmt: on
        ]
        for td, expt in test_cases:
            for data in (td, np.timedelta64(td)):
                self.assertEqualEscape(expt, data)

        self.log_ended(test)

    def test_escape_bytes(self) -> None:
        test = "ESCAPE BYTES"
        self.log_start(test)

        test_cases = [
            # 1. Trivial / baseline
            (b"", "_binary''"),
            (b"simple", "_binary'simple'"),
            # 2. Quotes
            (b"O'Reilly", "_binary'O\\'Reilly'"),  # single quote in middle
            (b'He said "hi"', "_binary'He said \\\"hi\\\"'"),
            (b"'", "_binary'\\''"),  # bare single quote
            (b'"', "_binary'\\\"'"),  # bare double quote
            # 3. Backslash
            (b"back\\slash", "_binary'back\\\\slash'"),
            # 4. Newline / carriage return
            (b"line1\nline2", "_binary'line1\\nline2'"),
            (b"carriage\rreturn", "_binary'carriage\\rreturn'"),
            # 5. NUL and Ctrl-Z
            (b"null\0byte", "_binary'null\\0byte'"),
            (b"ctrl_z\x1aend", "_binary'ctrl_z\\Zend'"),
            (b"\x00", "_binary'\\0'"),  # NUL only
            (b"\x1a", "_binary'\\Z'"),  # ASCII 26 only
            # 6. Combined specials
            (b"mix '\"\\\n\r\x1a end", "_binary'mix \\'\\\"\\\\\\n\\r\\Z end'"),
            # 8. Other ASCII that should *not* be touched
            (b"tab\ttest", "_binary'tab\ttest'"),  # \t is unchanged
            (b"space and !@#", "_binary'space and !@#'"),
            (b"control\x07bell", "_binary'control\x07bell'"),  # BEL (0x07) unchanged
            # 9. Long string
            (b"a" * 1000, "_binary'" + "a" * 1000 + "'"),
        ]
        for data, expt in test_cases:
            for dtype in (bytes, bytearray, memoryview):
                self.assertEqualEscape(expt, dtype(data))

        self.log_ended(test)

    def test_escape_decimal(self) -> None:
        test = "ESCAPE DECIMAL"
        self.log_start(test)

        for expt in ("-2.2345", "-1.2345", "0.0", "1.2345", "2.2345"):
            self.assertEqualEscape(expt, decimal.Decimal(expt))

        for i in ("nan", "-nan", "inf", "-inf"):
            with self.assertRaises(errors.EscapeValueError):
                escape(decimal.Decimal(i))

        self.log_ended(test)

    def test_escape_list_and_tuple(self) -> None:
        test = "ESCAPE LIST/TUPLE"
        self.log_start(test)

        for dtype in (list, tuple):
            # . flat
            arr = ["str", 1, 1.1]
            exp1 = "('str',1,1.1)"  # literal
            exp2 = ("'str'", "1", "1.1")  # itemize
            exp3 = ["'str'", "1", "1.1"]  # many
            data = dtype(arr)
            self.assertEqualEscape(exp1, data, many=False, itemize=False)
            self.assertEqualEscape(exp2, data, many=False, itemize=True)
            self.assertEqualEscape(exp3, data, many=True, itemize=True)

            # . nested
            arr = [["str1", 1, 1.1], ["str2", 2, 2.2]]
            exp1 = "('str1',1,1.1),('str2',2,2.2)"  # literal
            exp2 = ("('str1',1,1.1)", "('str2',2,2.2)")  # itemize
            exp3 = [("'str1'", "1", "1.1"), ("'str2'", "2", "2.2")]  # many
            data = dtype(dtype(i) for i in arr)
            self.assertEqualEscape(exp1, data, many=False, itemize=False)
            self.assertEqualEscape(exp2, data, many=False, itemize=True)
            self.assertEqualEscape(exp3, data, many=True, itemize=True)

            # . empty
            data = dtype()
            self.assertEqualEscape("()", data, many=False, itemize=False)
            self.assertEqualEscape((), data, many=False, itemize=True)
            self.assertEqualEscape([], data, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_set_and_frozenset(self) -> None:
        test = "ESCAPE SET/FROZENSET"
        self.log_start(test)

        for dtype in (set, frozenset):
            # . flat
            arr = [1, 2, 3]
            data = dtype(arr)
            exp1 = "(" + ",".join(str(i) for i in data) + ")"
            exp2 = tuple(str(i) for i in data)
            exp3 = [str(i) for i in data]
            self.assertEqualEscape(exp1, data, many=False, itemize=False)
            self.assertEqualEscape(exp2, data, many=False, itemize=True)
            self.assertEqualEscape(exp3, data, many=True, itemize=True)

            # . nested
            arr = [(1, 2, 3), (4, 5, 6)]
            data = dtype(arr)
            exp1 = ",".join("(" + ",".join(str(i) for i in v) + ")" for v in data)
            exp2 = tuple("(" + ",".join(str(i) for i in v) + ")" for v in data)
            exp3 = [tuple(str(i) for i in v) for v in data]
            self.assertEqualEscape(exp1, data, many=False, itemize=False)
            self.assertEqualEscape(exp2, data, many=False, itemize=True)
            self.assertEqualEscape(exp3, data, many=True, itemize=True)

            # . empty
            data = dtype()
            self.assertEqualEscape("()", data, many=False, itemize=False)
            self.assertEqualEscape((), data, many=False, itemize=True)
            self.assertEqualEscape([], data, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_range(self) -> None:
        test = "ESCAPE RANGE"
        self.log_start(test)

        for data in (range(11), range(-5, 6)):
            exp1 = "(" + ",".join(str(i) for i in data) + ")"
            exp2 = tuple(str(i) for i in data)
            exp3 = [str(i) for i in data]
            self.assertEqualEscape(exp1, data, many=False, itemize=False)
            self.assertEqualEscape(exp2, data, many=False, itemize=True)
            self.assertEqualEscape(exp3, data, many=True, itemize=True)

        # empty
        data = range(0)
        self.assertEqualEscape("()", data, many=False, itemize=False)
        self.assertEqualEscape((), data, many=False, itemize=True)
        self.assertEqualEscape([], data, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_sequence(self) -> None:
        test = "ESCAPE SEQUENCE"
        self.log_start(test)

        # fmt: off
        # . flat (keys)
        data = {"k1": "str", "k2": 1, "k3": 1.1}
        self.assertEqualEscape("('k1','k2','k3')", data.keys(), many=False, itemize=False)
        self.assertEqualEscape(("'k1'", "'k2'", "'k3'"), data.keys(), many=False, itemize=True)
        self.assertEqualEscape(["'k1'", "'k2'", "'k3'"], data.keys(), many=True, itemize=True)

        # . flat (values)
        self.assertEqualEscape("('str',1,1.1)", data.values(), many=False, itemize=False)
        self.assertEqualEscape(("'str'", "1", "1.1"), data.values(), many=False, itemize=True)
        self.assertEqualEscape(["'str'", "1", "1.1"], data.values(), many=True, itemize=True)

        # . nested (values)
        data = {"k1": ["str1", 1, 1.1], "k2": ["str2", 2, 2.2]}
        self.assertEqualEscape("('str1',1,1.1),('str2',2,2.2)", data.values(), many=False, itemize=False)
        self.assertEqualEscape(("('str1',1,1.1)", "('str2',2,2.2)"), data.values(), many=False, itemize=True)
        self.assertEqualEscape([("'str1'", "1", "1.1"), ("'str2'", "2", "2.2")], data.values(), many=True, itemize=True)
        # fmt: on

        # . empty (keys & values)
        for data in ({}.keys(), {}.values()):
            self.assertEqualEscape("()", data, many=False, itemize=False)
            self.assertEqualEscape((), data, many=False, itemize=True)
            self.assertEqualEscape([], data, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_dict(self) -> None:
        from collections import OrderedDict

        test = "ESCAPE DICT"
        self.log_start(test)

        # . flat
        dic = {"k1": "str", "k2": 1, "k3": 1.1}
        exp1 = "('str',1,1.1)"  # literal
        exp2 = ("'str'", "1", "1.1")  # itemize
        exp3 = ["'str'", "1", "1.1"]  # many
        for data in (dic, dic.values(), dic.items(), OrderedDict(dic)):
            self.assertEqualEscape(exp1, data, many=False, itemize=False)
            self.assertEqualEscape(exp2, data, many=False, itemize=True)
            self.assertEqualEscape(exp3, data, many=True, itemize=True)

        # . nested
        dic = {"k1": ["str1", 1, 1.1], "k2": ["str2", 2, 2.2]}
        exp1 = "('str1',1,1.1),('str2',2,2.2)"  # literal
        exp2 = ("('str1',1,1.1)", "('str2',2,2.2)")  # itemize
        exp3 = [("'str1'", "1", "1.1"), ("'str2'", "2", "2.2")]  # many
        for data in (dic, dic.values(), dic.items(), OrderedDict(dic)):
            self.assertEqualEscape(exp1, data, many=False, itemize=False)
            self.assertEqualEscape(exp2, data, many=False, itemize=True)
            self.assertEqualEscape(exp3, data, many=True, itemize=True)

        # . empty
        self.assertEqualEscape("()", {}, many=False, itemize=False)
        self.assertEqualEscape((), {}, many=False, itemize=True)
        self.assertEqualEscape([], {}, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_object(self) -> None:
        test = "ESCAPE ARRAY[OBJECT]"
        self.log_start(test)

        for dtype in ("O", object):
            # . 1-dimensional
            arr1 = [1, 1.1, "str"]
            exp1 = "(1,1.1,'str')"  # literal
            exp2 = ("1", "1.1", "'str'")  # itemize
            exp3 = ["1", "1.1", "'str'"]  # many
            for arr_1d in (np.array(arr1, dtype=dtype), pd.Series(arr1, dtype=dtype)):
                self.assertEqualEscape(exp1, arr_1d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_1d, many=False, itemize=True)
                self.assertEqualEscape(exp3, arr_1d, many=True, itemize=True)

            # . empty 1-dimensional
            for arr_1d in (np.array([], dtype=dtype), pd.Series([], dtype=dtype)):
                self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
                self.assertEqualEscape((), arr_1d, many=False, itemize=True)
                self.assertEqualEscape([], arr_1d, many=True, itemize=True)

            # . 2-dimensional
            arr2 = [arr1, arr1]
            exp1 = "(1,1.1,'str'),(1,1.1,'str')"  # literal
            exp2 = [("1", "1.1", "'str'"), ("1", "1.1", "'str'")]  # itemize / many
            for arr_2d in (
                np.array(arr2, dtype=dtype),
                pd.DataFrame(arr2, dtype=dtype),
            ):
                self.assertEqualEscape(exp1, arr_2d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_2d, many=False, itemize=True)
                self.assertEqualEscape(exp2, arr_2d, many=True, itemize=True)

            # . empty 2-dimensional
            for arr_2d in (
                np.array([[], []], dtype=dtype),
                pd.DataFrame([[], []], dtype=dtype),
            ):
                self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
                self.assertEqualEscape([], arr_2d, many=False, itemize=True)
                self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_int(self) -> None:
        test = "ESCAPE ARRAY[INT]"
        self.log_start(test)

        for dtype in (np.int8, np.int16, np.int32, np.int64):
            # . 1-dimensional
            arr1 = [i for i in range(-10, 11)]
            exp1 = "(" + ",".join(str(i) for i in arr1) + ")"  # literal
            exp2 = tuple(str(i) for i in arr1)  # itemize
            exp3 = [str(i) for i in arr1]  # many
            for arr_1d in (np.array(arr1, dtype=dtype), pd.Series(arr1, dtype=dtype)):
                self.assertEqualEscape(exp1, arr_1d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_1d, many=False, itemize=True)
                self.assertEqualEscape(exp3, arr_1d, many=True, itemize=True)

            # . empty 1-dimensional
            for arr_1d in (np.array([], dtype=dtype), pd.Series([], dtype=dtype)):
                self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
                self.assertEqualEscape((), arr_1d, many=False, itemize=True)
                self.assertEqualEscape([], arr_1d, many=True, itemize=True)

            # . 2-dimensional
            arr2 = [arr1, arr1]
            exp1 = ",".join("(" + ",".join(str(i) for i in row) + ")" for row in arr2)
            exp2 = [tuple(str(i) for i in row) for row in arr2]  # itemize / many
            for arr_2d in (
                np.array(arr2, dtype=dtype),
                pd.DataFrame(arr2, dtype=dtype),
            ):
                self.assertEqualEscape(exp1, arr_2d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_2d, many=False, itemize=True)
                self.assertEqualEscape(exp2, arr_2d, many=True, itemize=True)

            # . empty 2-dimensional
            for arr_2d in (
                np.array([[], []], dtype=dtype),
                pd.DataFrame([[], []], dtype=dtype),
            ):
                self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
                self.assertEqualEscape([], arr_2d, many=False, itemize=True)
                self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_uint(self) -> None:
        test = "ESCAPE ARRAY[UINT]"
        self.log_start(test)

        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            # . 1-dimensional
            arr1 = [i for i in range(11)]
            exp1 = "(" + ",".join(str(i) for i in arr1) + ")"  # literal
            exp2 = tuple(str(i) for i in arr1)  # itemize
            exp3 = [str(i) for i in arr1]  # many
            for arr_1d in (np.array(arr1, dtype=dtype), pd.Series(arr1, dtype=dtype)):
                self.assertEqualEscape(exp1, arr_1d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_1d, many=False, itemize=True)
                self.assertEqualEscape(exp3, arr_1d, many=True, itemize=True)

            # . empty 1-dimensional
            for arr_1d in (np.array([], dtype=dtype), pd.Series([], dtype=dtype)):
                self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
                self.assertEqualEscape((), arr_1d, many=False, itemize=True)
                self.assertEqualEscape([], arr_1d, many=True, itemize=True)

            # . 2-dimensional
            arr2 = [arr1, arr1]
            exp1 = ",".join("(" + ",".join(str(i) for i in row) + ")" for row in arr2)
            exp2 = [tuple(str(i) for i in row) for row in arr2]  # itemize / many
            for arr_2d in (
                np.array(arr2, dtype=dtype),
                pd.DataFrame(arr2, dtype=dtype),
            ):
                self.assertEqualEscape(exp1, arr_2d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_2d, many=False, itemize=True)
                self.assertEqualEscape(exp2, arr_2d, many=True, itemize=True)

            # . empty 2-dimensional
            for arr_2d in (
                np.array([[], []], dtype=dtype),
                pd.DataFrame([[], []], dtype=dtype),
            ):
                self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
                self.assertEqualEscape([], arr_2d, many=False, itemize=True)
                self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_float(self) -> None:
        test = "ESCAPE ARRAY[FLOAT]"
        self.log_start(test)

        for dtype in (np.float32, np.float64):
            # . 1-dimensional
            arr1 = [-2.2, -1.1, 0.0, 1.1, 2.2]
            exp1 = "(" + ",".join(str(i) for i in arr1) + ")"  # literal
            exp2 = tuple(str(i) for i in arr1)  # itemize
            exp3 = [str(i) for i in arr1]  # many
            for arr_1d in (np.array(arr1, dtype=dtype), pd.Series(arr1, dtype=dtype)):
                self.assertEqualEscape(exp1, arr_1d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_1d, many=False, itemize=True)
                self.assertEqualEscape(exp3, arr_1d, many=True, itemize=True)

            # . empty 1-dimensional
            for arr_1d in (np.array([], dtype=dtype), pd.Series([], dtype=dtype)):
                self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
                self.assertEqualEscape((), arr_1d, many=False, itemize=True)
                self.assertEqualEscape([], arr_1d, many=True, itemize=True)

            # . 2-dimensional
            arr2 = [arr1, arr1]
            exp1 = ",".join("(" + ",".join(str(i) for i in row) + ")" for row in arr2)
            exp2 = [tuple(str(i) for i in row) for row in arr2]  # itemize / many
            for arr_2d in (
                np.array(arr2, dtype=dtype),
                pd.DataFrame(arr2, dtype=dtype),
            ):
                self.assertEqualEscape(exp1, arr_2d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_2d, many=False, itemize=True)
                self.assertEqualEscape(exp2, arr_2d, many=True, itemize=True)

            # . empty 2-dimensional
            for arr_2d in (
                np.array([[], []], dtype=dtype),
                pd.DataFrame([[], []], dtype=dtype),
            ):
                self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
                self.assertEqualEscape([], arr_2d, many=False, itemize=True)
                self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_bool(self) -> None:
        test = "ESCAPE ARRAY[BOOL]"
        self.log_start(test)

        for dtype in (np.bool_, bool):
            # . 1-dimensional
            arr1 = [True, False, True]
            exp1 = "(1,0,1)"  # literal
            exp2 = ("1", "0", "1")  # itemize
            exp3 = ["1", "0", "1"]  # many
            for arr_1d in (np.array(arr1, dtype=dtype), pd.Series(arr1, dtype=dtype)):
                self.assertEqualEscape(exp1, arr_1d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_1d, many=False, itemize=True)
                self.assertEqualEscape(exp3, arr_1d, many=True, itemize=True)

            # . empty 1-dimensional
            for arr_1d in (np.array([], dtype=dtype), pd.Series([], dtype=dtype)):
                self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
                self.assertEqualEscape((), arr_1d, many=False, itemize=True)
                self.assertEqualEscape([], arr_1d, many=True, itemize=True)

            # . 2-dimensional
            arr2 = [arr1, arr1]
            exp1 = "(1,0,1),(1,0,1)"  # literal
            exp2 = [("1", "0", "1"), ("1", "0", "1")]  # itemize / many
            for arr_2d in (
                np.array(arr2, dtype=dtype),
                pd.DataFrame(arr2, dtype=dtype),
            ):
                self.assertEqualEscape(exp1, arr_2d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_2d, many=False, itemize=True)
                self.assertEqualEscape(exp2, arr_2d, many=True, itemize=True)

            # . empty 2-dimensional
            for arr_2d in (
                np.array([[], []], dtype=dtype),
                pd.DataFrame([[], []], dtype=dtype),
            ):
                self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
                self.assertEqualEscape([], arr_2d, many=False, itemize=True)
                self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_dt64(self) -> None:
        from cytimes import Pddt

        test = "ESCAPE ARRAY[DATETIME64]"
        self.log_start(test)

        # fmt: off
        # . 1-dimensional: datetime64[D]
        arr1 = [-1, 0, 1]
        arr_1d = np.array(arr1, dtype="datetime64[D]")
        self.assertEqualEscape("('1969-12-31 00:00:00','1970-01-01 00:00:00','1970-01-02 00:00:00')", arr_1d, many=False, itemize=False)
        self.assertEqualEscape(("'1969-12-31 00:00:00'", "'1970-01-01 00:00:00'", "'1970-01-02 00:00:00'"), arr_1d, many=False, itemize=True)
        self.assertEqualEscape(["'1969-12-31 00:00:00'", "'1970-01-01 00:00:00'", "'1970-01-02 00:00:00'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: datetime64[h]
        arr_1d = np.array(arr1, dtype="datetime64[h]")
        self.assertEqualEscape("('1969-12-31 23:00:00','1970-01-01 00:00:00','1970-01-01 01:00:00')", arr_1d, many=False, itemize=False)
        self.assertEqualEscape(("'1969-12-31 23:00:00'", "'1970-01-01 00:00:00'", "'1970-01-01 01:00:00'"), arr_1d, many=False, itemize=True)
        self.assertEqualEscape(["'1969-12-31 23:00:00'", "'1970-01-01 00:00:00'", "'1970-01-01 01:00:00'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: datetime64[m]
        arr_1d = np.array(arr1, dtype="datetime64[m]")
        self.assertEqualEscape("('1969-12-31 23:59:00','1970-01-01 00:00:00','1970-01-01 00:01:00')", arr_1d, many=False, itemize=False)
        self.assertEqualEscape(("'1969-12-31 23:59:00'", "'1970-01-01 00:00:00'", "'1970-01-01 00:01:00'"), arr_1d, many=False, itemize=True)
        self.assertEqualEscape(["'1969-12-31 23:59:00'", "'1970-01-01 00:00:00'", "'1970-01-01 00:01:00'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: datetime64[s]
        for arr_1d in (
            np.array(arr1, dtype="datetime64[s]"), 
            pd.Series(np.array(arr1, dtype="datetime64[s]")), 
            pd.DatetimeIndex(arr1, dtype="datetime64[s]"),
            Pddt(pd.DatetimeIndex(arr1, dtype="datetime64[s]")),
        ):
            self.assertEqualEscape("('1969-12-31 23:59:59','1970-01-01 00:00:00','1970-01-01 00:00:01')", arr_1d, many=False, itemize=False)
            self.assertEqualEscape(("'1969-12-31 23:59:59'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:01'"), arr_1d, many=False, itemize=True)
            self.assertEqualEscape(["'1969-12-31 23:59:59'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:01'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: datetime64[ms]
        for arr_1d in (
            np.array(arr1, dtype="datetime64[ms]"), 
            pd.Series(np.array(arr1, dtype="datetime64[ms]")), 
            pd.DatetimeIndex(arr1, dtype="datetime64[ms]"),
            Pddt(pd.DatetimeIndex(arr1, dtype="datetime64[ms]")),
        ):
            self.assertEqualEscape("('1969-12-31 23:59:59.999000','1970-01-01 00:00:00','1970-01-01 00:00:00.001000')", arr_1d, many=False, itemize=False)
            self.assertEqualEscape(("'1969-12-31 23:59:59.999000'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.001000'"), arr_1d, many=False, itemize=True)
            self.assertEqualEscape(["'1969-12-31 23:59:59.999000'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.001000'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: datetime64[us]
        for arr_1d in (
            np.array(arr1, dtype="datetime64[us]"), 
            pd.Series(np.array(arr1, dtype="datetime64[us]")), 
            pd.DatetimeIndex(arr1, dtype="datetime64[us]"),
            Pddt(pd.DatetimeIndex(arr1, dtype="datetime64[us]")),
        ):
            self.assertEqualEscape("('1969-12-31 23:59:59.999999','1970-01-01 00:00:00','1970-01-01 00:00:00.000001')", arr_1d, many=False, itemize=False)
            self.assertEqualEscape(("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.000001'"), arr_1d, many=False, itemize=True)
            self.assertEqualEscape(["'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.000001'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: datetime64[ns]
        for arr_1d in (
            np.array(arr1, dtype="datetime64[ns]"), 
            pd.Series(np.array(arr1, dtype="datetime64[ns]")), 
            pd.DatetimeIndex(arr1, dtype="datetime64[ns]"),
            Pddt(pd.DatetimeIndex(arr1, dtype="datetime64[ns]")),
        ):
            self.assertEqualEscape("('1969-12-31 23:59:59.999999','1970-01-01 00:00:00','1970-01-01 00:00:00')", arr_1d, many=False, itemize=False)
            self.assertEqualEscape(("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00'"), arr_1d, many=False, itemize=True)
            self.assertEqualEscape(["'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00'"], arr_1d, many=True, itemize=True)
        # fmt: on

        # . 1-dimensional: empty
        for unit in ("D", "h", "m", "s", "ms", "us", "ns"):
            arr_1d = np.array([], dtype=f"datetime64[{unit}]")
            self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
            self.assertEqualEscape((), arr_1d, many=False, itemize=True)
            self.assertEqualEscape([], arr_1d, many=True, itemize=True)
            try:
                arr_1d = pd.Series([], dtype=f"datetime64[{unit}]")
            except TypeError:
                continue
            self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
            self.assertEqualEscape((), arr_1d, many=False, itemize=True)
            self.assertEqualEscape([], arr_1d, many=True, itemize=True)
            try:
                arr_1d = pd.DatetimeIndex([], dtype=f"datetime64[{unit}]")
            except TypeError:
                continue
            self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
            self.assertEqualEscape((), arr_1d, many=False, itemize=True)
            self.assertEqualEscape([], arr_1d, many=True, itemize=True)
            try:
                arr_1d = Pddt(pd.DatetimeIndex([], dtype=f"datetime64[{unit}]"))
            except TypeError:
                continue
            self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
            self.assertEqualEscape((), arr_1d, many=False, itemize=True)
            self.assertEqualEscape([], arr_1d, many=True, itemize=True)

        # fmt: off
        # . 2-dimensional: datetime64[d]
        arr2 = [arr1, arr1]
        arr_2d = np.array(arr2, dtype="datetime64[D]")
        self.assertEqualEscape( 
            "('1969-12-31 00:00:00','1970-01-01 00:00:00','1970-01-02 00:00:00'),"
            "('1969-12-31 00:00:00','1970-01-01 00:00:00','1970-01-02 00:00:00')", 
        arr_2d,
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'1969-12-31 00:00:00'", "'1970-01-01 00:00:00'", "'1970-01-02 00:00:00'"),
             ("'1969-12-31 00:00:00'", "'1970-01-01 00:00:00'", "'1970-01-02 00:00:00'")], 
            arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'1969-12-31 00:00:00'", "'1970-01-01 00:00:00'", "'1970-01-02 00:00:00'"),
             ("'1969-12-31 00:00:00'", "'1970-01-01 00:00:00'", "'1970-01-02 00:00:00'")], 
            arr_2d,
            many=True, itemize=True)

        # . 2-dimensional: datetime64[h]
        arr_2d = np.array(arr2, dtype="datetime64[h]")
        self.assertEqualEscape( 
            "('1969-12-31 23:00:00','1970-01-01 00:00:00','1970-01-01 01:00:00'),"
            "('1969-12-31 23:00:00','1970-01-01 00:00:00','1970-01-01 01:00:00')", 
        arr_2d,
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'1969-12-31 23:00:00'", "'1970-01-01 00:00:00'", "'1970-01-01 01:00:00'"),
             ("'1969-12-31 23:00:00'", "'1970-01-01 00:00:00'", "'1970-01-01 01:00:00'")], 
            arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'1969-12-31 23:00:00'", "'1970-01-01 00:00:00'", "'1970-01-01 01:00:00'"),
             ("'1969-12-31 23:00:00'", "'1970-01-01 00:00:00'", "'1970-01-01 01:00:00'")], 
            arr_2d,
            many=True, itemize=True)
        
        # . 2-dimensional: datetime64[m]
        arr_2d = np.array(arr2, dtype="datetime64[m]")
        self.assertEqualEscape( 
            "('1969-12-31 23:59:00','1970-01-01 00:00:00','1970-01-01 00:01:00'),"
            "('1969-12-31 23:59:00','1970-01-01 00:00:00','1970-01-01 00:01:00')", 
        arr_2d,
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'1969-12-31 23:59:00'", "'1970-01-01 00:00:00'", "'1970-01-01 00:01:00'"),
             ("'1969-12-31 23:59:00'", "'1970-01-01 00:00:00'", "'1970-01-01 00:01:00'")], 
            arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'1969-12-31 23:59:00'", "'1970-01-01 00:00:00'", "'1970-01-01 00:01:00'"),
             ("'1969-12-31 23:59:00'", "'1970-01-01 00:00:00'", "'1970-01-01 00:01:00'")], 
            arr_2d,
            many=True, itemize=True)
        
        # . 2-dimensional: datetime64[s]
        for arr_2d in (np.array(arr2, dtype="datetime64[s]"), pd.DataFrame(arr2, dtype="datetime64[s]")):
            self.assertEqualEscape(
                "('1969-12-31 23:59:59','1970-01-01 00:00:00','1970-01-01 00:00:01'),"
                "('1969-12-31 23:59:59','1970-01-01 00:00:00','1970-01-01 00:00:01')", 
                arr_2d,
                many=False, itemize=False)
            self.assertEqualEscape(
                [("'1969-12-31 23:59:59'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:01'"),
                ("'1969-12-31 23:59:59'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:01'")], 
                arr_2d,
                many=False, itemize=True)
            self.assertEqualEscape(
                [("'1969-12-31 23:59:59'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:01'"),
                ("'1969-12-31 23:59:59'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:01'")], 
                arr_2d,
                many=True, itemize=True)
        
        # . 2-dimensional: datetime64[ms]
        for arr_2d in (np.array(arr2, dtype="datetime64[ms]"), pd.DataFrame(arr2, dtype="datetime64[ms]")):
            self.assertEqualEscape(
                "('1969-12-31 23:59:59.999000','1970-01-01 00:00:00','1970-01-01 00:00:00.001000'),"
                "('1969-12-31 23:59:59.999000','1970-01-01 00:00:00','1970-01-01 00:00:00.001000')", 
                arr_2d, 
                many=False, itemize=False)
            self.assertEqualEscape(
                [("'1969-12-31 23:59:59.999000'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.001000'"),
                ("'1969-12-31 23:59:59.999000'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.001000'")], 
                arr_2d,
                many=False, itemize=True)
            self.assertEqualEscape(
                [("'1969-12-31 23:59:59.999000'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.001000'"),
                ("'1969-12-31 23:59:59.999000'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.001000'")], 
                arr_2d,
                many=True, itemize=True)
        
        # . 2-dimensional: datetime64[us]
        for arr_2d in (np.array(arr2, dtype="datetime64[us]"), pd.DataFrame(arr2, dtype="datetime64[us]")):
            self.assertEqualEscape(
                "('1969-12-31 23:59:59.999999','1970-01-01 00:00:00','1970-01-01 00:00:00.000001'),"
                "('1969-12-31 23:59:59.999999','1970-01-01 00:00:00','1970-01-01 00:00:00.000001')", 
                arr_2d, 
                many=False, itemize=False)
            self.assertEqualEscape(
                [("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.000001'"),
                ("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.000001'")], 
                arr_2d,
                many=False, itemize=True)
            self.assertEqualEscape(
                [("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.000001'"),
                ("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00.000001'")], 
                arr_2d,
                many=True, itemize=True)
        
        # . 2-dimensional: datetime64[ns]
        for arr_2d in (np.array(arr2, dtype="datetime64[ns]"), pd.DataFrame(arr2, dtype="datetime64[ns]")):
            self.assertEqualEscape(
                "('1969-12-31 23:59:59.999999','1970-01-01 00:00:00','1970-01-01 00:00:00'),"
                "('1969-12-31 23:59:59.999999','1970-01-01 00:00:00','1970-01-01 00:00:00')", 
                arr_2d, 
                many=False, itemize=False)
            self.assertEqualEscape(
                [("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00'"),
                ("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00'")], 
                arr_2d,
                many=False, itemize=True)
            self.assertEqualEscape(
                [("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00'"),
                ("'1969-12-31 23:59:59.999999'", "'1970-01-01 00:00:00'", "'1970-01-01 00:00:00'")], 
                arr_2d,
                many=True, itemize=True)
        # fmt: on

        # . 2-dimensional: empty
        for unit in ("D", "h", "m", "s", "ms", "us", "ns"):
            arr_2d = np.array([[], []], dtype=f"datetime64[{unit}]")
            self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
            self.assertEqualEscape([], arr_2d, many=False, itemize=True)
            self.assertEqualEscape([], arr_2d, many=True, itemize=True)
            try:
                arr_2d = pd.DataFrame([[], []], dtype=f"datetime64[{unit}]")
            except TypeError:
                continue
            self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
            self.assertEqualEscape([], arr_2d, many=False, itemize=True)
            self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_td64(self) -> None:
        test = "ESCAPE ARRAY[TIMESPAN64]"
        self.log_start(test)

        # fmt: off
        # . 1-dimensional: timedelta64[D]
        arr1 = [-1, 0, 1]
        arr_1d = np.array(arr1, dtype="timedelta64[D]")
        self.assertEqualEscape("('-24:00:00','00:00:00','24:00:00')", arr_1d, many=False, itemize=False)
        self.assertEqualEscape(("'-24:00:00'", "'00:00:00'", "'24:00:00'"), arr_1d, many=False, itemize=True)
        self.assertEqualEscape(["'-24:00:00'", "'00:00:00'", "'24:00:00'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: timedelta64[h]
        arr_1d = np.array(arr1, dtype="timedelta64[h]")
        self.assertEqualEscape("('-01:00:00','00:00:00','01:00:00')", arr_1d, many=False, itemize=False)
        self.assertEqualEscape(("'-01:00:00'", "'00:00:00'", "'01:00:00'"), arr_1d, many=False, itemize=True)
        self.assertEqualEscape(["'-01:00:00'", "'00:00:00'", "'01:00:00'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: timedelta64[m]
        arr_1d = np.array(arr1, dtype="timedelta64[m]")
        self.assertEqualEscape("('-00:01:00','00:00:00','00:01:00')", arr_1d, many=False, itemize=False)
        self.assertEqualEscape(("'-00:01:00'", "'00:00:00'", "'00:01:00'"), arr_1d, many=False, itemize=True)
        self.assertEqualEscape(["'-00:01:00'", "'00:00:00'", "'00:01:00'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: timedelta64[s]
        for arr_1d in (np.array(arr1, dtype="timedelta64[s]"), pd.Series(np.array(arr1, dtype="timedelta64[s]"))):
            self.assertEqualEscape("('-00:00:01','00:00:00','00:00:01')", arr_1d, many=False, itemize=False)
            self.assertEqualEscape(("'-00:00:01'", "'00:00:00'", "'00:00:01'"), arr_1d, many=False, itemize=True)
            self.assertEqualEscape(["'-00:00:01'", "'00:00:00'", "'00:00:01'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: timedelta64[ms]
        for arr_1d in (np.array(arr1, dtype="timedelta64[ms]"), pd.Series(np.array(arr1, dtype="timedelta64[ms]"))):
            self.assertEqualEscape("('-00:00:00.001000','00:00:00','00:00:00.001000')", arr_1d, many=False, itemize=False)
            self.assertEqualEscape(("'-00:00:00.001000'", "'00:00:00'", "'00:00:00.001000'"), arr_1d, many=False, itemize=True)
            self.assertEqualEscape(["'-00:00:00.001000'", "'00:00:00'", "'00:00:00.001000'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: timedelta64[us]
        for arr_1d in (np.array(arr1, dtype="timedelta64[us]"), pd.Series(np.array(arr1, dtype="timedelta64[us]"))):
            self.assertEqualEscape("('-00:00:00.000001','00:00:00','00:00:00.000001')", arr_1d, many=False, itemize=False)
            self.assertEqualEscape(("'-00:00:00.000001'", "'00:00:00'", "'00:00:00.000001'"), arr_1d, many=False, itemize=True)
            self.assertEqualEscape(["'-00:00:00.000001'", "'00:00:00'", "'00:00:00.000001'"], arr_1d, many=True, itemize=True)

        # . 1-dimensional: timedelta64[ns]
        for arr_1d in (np.array(arr1, dtype="timedelta64[ns]"), pd.Series(np.array(arr1, dtype="timedelta64[ns]"))):
            self.assertEqualEscape("('-00:00:00.000001','00:00:00','00:00:00')", arr_1d, many=False, itemize=False)
            self.assertEqualEscape(("'-00:00:00.000001'", "'00:00:00'", "'00:00:00'"), arr_1d, many=False, itemize=True)
            self.assertEqualEscape(["'-00:00:00.000001'", "'00:00:00'", "'00:00:00'"], arr_1d, many=True, itemize=True)
        # fmt: on

        # . 1-dimensional: empty
        for unit in ("D", "h", "m", "s", "ms", "us", "ns"):
            arr_1d = np.array([], dtype=f"timedelta64[{unit}]")
            self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
            self.assertEqualEscape((), arr_1d, many=False, itemize=True)
            self.assertEqualEscape([], arr_1d, many=True, itemize=True)
            try:
                arr_1d = pd.Series([], dtype=f"timedelta64[{unit}]")
            except TypeError:
                continue
            self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
            self.assertEqualEscape((), arr_1d, many=False, itemize=True)
            self.assertEqualEscape([], arr_1d, many=True, itemize=True)
            try:
                arr_1d = pd.TimedeltaIndex([], dtype=f"timedelta64[{unit}]")
            except TypeError:
                continue
            self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
            self.assertEqualEscape((), arr_1d, many=False, itemize=True)
            self.assertEqualEscape([], arr_1d, many=True, itemize=True)

        # fmt: off
        # . 2-dimensional: timedelta64[D]
        arr2 = [arr1, arr1]
        arr_2d = np.array(arr2, dtype="timedelta64[D]")
        self.assertEqualEscape(
            "('-24:00:00','00:00:00','24:00:00'),"
            "('-24:00:00','00:00:00','24:00:00')", 
            arr_2d,
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'-24:00:00'", "'00:00:00'", "'24:00:00'"),
             ("'-24:00:00'", "'00:00:00'", "'24:00:00'")], 
             arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'-24:00:00'", "'00:00:00'", "'24:00:00'"),
             ("'-24:00:00'", "'00:00:00'", "'24:00:00'")], 
             arr_2d,
            many=True, itemize=True)
        
        # . 2-dimensional: timedelta64[h]
        arr_2d = np.array(arr2, dtype="timedelta64[h]")
        self.assertEqualEscape(
            "('-01:00:00','00:00:00','01:00:00'),"
            "('-01:00:00','00:00:00','01:00:00')", 
            arr_2d, 
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'-01:00:00'", "'00:00:00'", "'01:00:00'"),
             ("'-01:00:00'", "'00:00:00'", "'01:00:00'")], 
             arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'-01:00:00'", "'00:00:00'", "'01:00:00'"),
             ("'-01:00:00'", "'00:00:00'", "'01:00:00'")], 
             arr_2d,
            many=True, itemize=True)
        
        # . 2-dimensional: timedelta64[m]
        arr_2d = np.array(arr2, dtype="timedelta64[m]")
        self.assertEqualEscape(
            "('-00:01:00','00:00:00','00:01:00'),"
            "('-00:01:00','00:00:00','00:01:00')", 
            arr_2d, 
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'-00:01:00'", "'00:00:00'", "'00:01:00'"),
             ("'-00:01:00'", "'00:00:00'", "'00:01:00'")], 
             arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'-00:01:00'", "'00:00:00'", "'00:01:00'"),
             ("'-00:01:00'", "'00:00:00'", "'00:01:00'")], 
             arr_2d,
            many=True, itemize=True)
        
        # . 2-dimensional: timedelta64[s]
        arr_2d = np.array(arr2, dtype="timedelta64[s]")
        self.assertEqualEscape(
            "('-00:00:01','00:00:00','00:00:01'),"
            "('-00:00:01','00:00:00','00:00:01')", 
            arr_2d, 
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'-00:00:01'", "'00:00:00'", "'00:00:01'"),
             ("'-00:00:01'", "'00:00:00'", "'00:00:01'")], 
             arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'-00:00:01'", "'00:00:00'", "'00:00:01'"),
             ("'-00:00:01'", "'00:00:00'", "'00:00:01'")], 
             arr_2d,
            many=True, itemize=True)
        
        # . 2-dimensional: timedelta64[ms]
        arr_2d = np.array(arr2, dtype="timedelta64[ms]")
        self.assertEqualEscape(
            "('-00:00:00.001000','00:00:00','00:00:00.001000'),"
            "('-00:00:00.001000','00:00:00','00:00:00.001000')", 
            arr_2d, 
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'-00:00:00.001000'", "'00:00:00'", "'00:00:00.001000'"),
             ("'-00:00:00.001000'", "'00:00:00'", "'00:00:00.001000'")], 
             arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'-00:00:00.001000'", "'00:00:00'", "'00:00:00.001000'"),
             ("'-00:00:00.001000'", "'00:00:00'", "'00:00:00.001000'")], 
             arr_2d,
            many=True, itemize=True)
        
        # . 2-dimensional: timedelta64[us]
        arr_2d = np.array(arr2, dtype="timedelta64[us]")
        self.assertEqualEscape(
            "('-00:00:00.000001','00:00:00','00:00:00.000001'),"
            "('-00:00:00.000001','00:00:00','00:00:00.000001')", 
            arr_2d, 
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'-00:00:00.000001'", "'00:00:00'", "'00:00:00.000001'"),
             ("'-00:00:00.000001'", "'00:00:00'", "'00:00:00.000001'")], 
             arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'-00:00:00.000001'", "'00:00:00'", "'00:00:00.000001'"),
             ("'-00:00:00.000001'", "'00:00:00'", "'00:00:00.000001'")], 
             arr_2d,
            many=True, itemize=True)
        
        # . 2-dimensional: timedelta64[ns]
        arr_2d = np.array(arr2, dtype="timedelta64[ns]")
        self.assertEqualEscape(
            "('-00:00:00.000001','00:00:00','00:00:00'),"
            "('-00:00:00.000001','00:00:00','00:00:00')", 
            arr_2d, 
            many=False, itemize=False)
        self.assertEqualEscape(
            [("'-00:00:00.000001'", "'00:00:00'", "'00:00:00'"),
             ("'-00:00:00.000001'", "'00:00:00'", "'00:00:00'")], 
             arr_2d,
            many=False, itemize=True)
        self.assertEqualEscape(
            [("'-00:00:00.000001'", "'00:00:00'", "'00:00:00'"),
             ("'-00:00:00.000001'", "'00:00:00'", "'00:00:00'")], 
             arr_2d,
            many=True, itemize=True)
        # fmt: on

        # . 2-dimensional: empty
        for unit in ("D", "h", "m", "s", "ms", "us", "ns"):
            arr_2d = np.array([[], []], dtype=f"timedelta64[{unit}]")
            self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
            self.assertEqualEscape([], arr_2d, many=False, itemize=True)
            self.assertEqualEscape([], arr_2d, many=True, itemize=True)
            try:
                arr_2d = pd.DataFrame([[], []], dtype=f"timedelta64[{unit}]")
            except TypeError:
                continue
            self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
            self.assertEqualEscape([], arr_2d, many=False, itemize=True)
            self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_bytes(self) -> None:
        test = "ESCAPE ARRAY[BYTES]"
        self.log_start(test)

        for dtype in ("S", np.bytes_):
            # . 1-dimensional
            arr1 = [b"abc", b"def", b"ghi"]
            exp1 = "(_binary'abc',_binary'def',_binary'ghi')"
            exp2 = ("_binary'abc'", "_binary'def'", "_binary'ghi'")
            exp3 = ["_binary'abc'", "_binary'def'", "_binary'ghi'"]
            for arr_1d in (
                np.array(arr1, dtype=dtype),
                pd.Series(np.array(arr1, dtype=dtype)),
            ):
                self.assertEqualEscape(exp1, arr_1d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_1d, many=False, itemize=True)
                self.assertEqualEscape(exp3, arr_1d, many=True, itemize=True)

            # . empty 1-dimensional
            for arr_1d in (
                np.array([], dtype=dtype),
                pd.Series(np.array([], dtype=dtype)),
            ):
                self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
                self.assertEqualEscape((), arr_1d, many=False, itemize=True)
                self.assertEqualEscape([], arr_1d, many=True, itemize=True)

            # . 2-dimensional
            arr2 = [arr1, arr1]
            exp1 = "(_binary'abc',_binary'def',_binary'ghi'),(_binary'abc',_binary'def',_binary'ghi')"
            exp2 = [
                ("_binary'abc'", "_binary'def'", "_binary'ghi'"),
                ("_binary'abc'", "_binary'def'", "_binary'ghi'"),
            ]
            for arr_2d in (
                np.array(arr2, dtype=dtype),
                pd.DataFrame(np.array(arr2, dtype=dtype)),
            ):
                self.assertEqualEscape(exp1, arr_2d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_2d, many=False, itemize=True)
                self.assertEqualEscape(exp2, arr_2d, many=True, itemize=True)

            # . empty 2-dimensional
            for arr_2d in (
                np.array([[], []], dtype=dtype),
                pd.DataFrame(np.array([[], []], dtype=dtype)),
            ):
                self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
                self.assertEqualEscape([], arr_2d, many=False, itemize=True)
                self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_array_unicode(self) -> None:
        test = "ESCAPE ARRAY[UNICODE]"
        self.log_start(test)

        for dtype in ("U", np.str_):
            # . 1-dimensional
            arr1 = ["abc", "def", "ghi"]
            exp1 = "('abc','def','ghi')"
            exp2 = ("'abc'", "'def'", "'ghi'")
            exp3 = ["'abc'", "'def'", "'ghi'"]
            for arr_1d in (
                np.array(arr1, dtype=dtype),
                pd.Series(np.array(arr1, dtype=dtype)),
            ):
                self.assertEqualEscape(exp1, arr_1d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_1d, many=False, itemize=True)
                self.assertEqualEscape(exp3, arr_1d, many=True, itemize=True)

            # . empty 1-dimensional
            for arr_1d in (
                np.array([], dtype=dtype),
                pd.Series(np.array([], dtype=dtype)),
            ):
                self.assertEqualEscape("()", arr_1d, many=False, itemize=False)
                self.assertEqualEscape((), arr_1d, many=False, itemize=True)
                self.assertEqualEscape([], arr_1d, many=True, itemize=True)

            # . 2-dimensional
            arr2 = [arr1, arr1]
            exp1 = "('abc','def','ghi'),('abc','def','ghi')"
            exp2 = [("'abc'", "'def'", "'ghi'"), ("'abc'", "'def'", "'ghi'")]
            for arr_2d in (
                np.array(arr2, dtype=dtype),
                pd.DataFrame(np.array(arr2, dtype=dtype)),
            ):
                self.assertEqualEscape(exp1, arr_2d, many=False, itemize=False)
                self.assertEqualEscape(exp2, arr_2d, many=False, itemize=True)
                self.assertEqualEscape(exp2, arr_2d, many=True, itemize=True)

            # . empty 2-dimensional
            for arr_2d in (
                np.array([[], []], dtype=dtype),
                pd.DataFrame(np.array([[], []], dtype=dtype)),
            ):
                self.assertEqualEscape("()", arr_2d, many=False, itemize=False)
                self.assertEqualEscape([], arr_2d, many=False, itemize=True)
                self.assertEqualEscape([], arr_2d, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_dataframe(self) -> None:
        test = "ESCAPE DATAFRAME"
        self.log_start(test)

        # . normal DataFrame
        data = pd.DataFrame(
            {"a": [1, 2, 3], "b": [1.1, 2.2, 3.3], "c": ["a", "b", "c"]}
        )
        exp1 = "(1,1.1,'a'),(2,2.2,'b'),(3,3.3,'c')"
        exp2 = [("1", "1.1", "'a'"), ("2", "2.2", "'b'"), ("3", "3.3", "'c'")]
        self.assertEqualEscape(exp1, data, many=False, itemize=False)
        self.assertEqualEscape(exp2, data, many=False, itemize=True)
        self.assertEqualEscape(exp2, data, many=True, itemize=True)

        # . empty DataFrame
        for df in (pd.DataFrame(), pd.DataFrame(columns=["a", "b", "c"])):
            self.assertEqualEscape("()", df, many=False, itemize=False)
            self.assertEqualEscape([], df, many=False, itemize=True)
            self.assertEqualEscape([], df, many=True, itemize=True)

        self.log_ended(test)

    def test_escape_custom(self) -> None:
        from sqlcycli.transcode import BIT, JSON

        test = "ESCAPE CUSTOM TYPES"
        self.log_start(test)

        # BIT: bytes
        for value, expt in (
            (b"\x01", "1"),
            (b"\x00\x00\x00\x17\xd8 D\x00", "102410241024"),
        ):
            for dtype in (bytes, bytearray, memoryview, np.bytes_):
                self.assertEqualEscape(expt, BIT(dtype(value)))

        # BIT: integer
        for value, expt in (
            (0, "0"),
            (10, "10"),
        ):
            for dtype in (
                int,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                self.assertEqualEscape(expt, BIT(dtype(value)))

        # BIT: invalid
        with self.assertRaises(errors.EscapeError):
            escape(BIT("apple"))  # not bytes or int

        # JSON
        for value, expt in (
            ({"a": 1, "b": 2}, '\'{\\"a\\":1,\\"b\\":2}\''),
            ([1, 1.1, "foo"], "'[1,1.1,\\\"foo\\\"]'"),
        ):
            self.assertEqualEscape(expt, JSON(value))

        # JSON: invalid
        with self.assertRaises(errors.EscapeError):
            escape(JSON(pd.Series([1, 2, 3])))

        self.log_ended(test)

    # Utils
    def assertEqualEscape(
        self,
        expt: str,
        data: object,
        many: bool | None = None,
        itemize: bool | None = None,
    ) -> None:
        # fmt: off
        if many is not None:
            many = bool(many)
            if itemize is not None:
                res = escape(data, many=many, itemize=bool(itemize))
                self.assertEqual(res, expt, f"escape({data!r}) >>> {res!r} != {expt!r}")
            else:
                for itemize in (True, False):
                    res = escape(data, many=many, itemize=itemize)
                    self.assertEqual(res, expt, f"escape({data!r}) >>> {res!r} != {expt!r}")

        elif itemize is not None:
            itemize = bool(itemize)
            for many in (True, False):
                res = escape(data, many=many, itemize=itemize)
                self.assertEqual(res, expt, f"escape({data!r}) >>> {res!r} != {expt!r}")

        else:
            for many in (True, False):
                for itemize in (True, False):
                    res = escape(data, many=many, itemize=itemize)
                    self.assertEqual(res, expt, f"escape({data!r}) >>> {res!r} != {expt!r}")
        # fmt: on


class TestDecode(TestCase):
    name: str = "Decode"

    def test_all(self) -> None:
        self.decode_string()
        self.decode_integer()
        self.decode_float()
        self.decode_decimal()
        self.decode_bit()
        self.decode_date()
        self.decode_datetime()
        self.decode_time()
        self.decode_enum()
        self.decode_set()
        self.decode_json()

    def decode_string(self) -> None:
        test = "DECODE STRING"
        self.log_start(test)

        # CHAR types
        for field_type in (
            FIELD_TYPE.STRING,
            FIELD_TYPE.VAR_STRING,
            FIELD_TYPE.VARCHAR,
            FIELD_TYPE.TINY_BLOB,
            FIELD_TYPE.BLOB,
            FIELD_TYPE.MEDIUM_BLOB,
            FIELD_TYPE.LONG_BLOB,
        ):
            self.assertEqualDecode("char", b"char", field_type, is_binary=False)

        # BINARY types
        for field_type in (
            FIELD_TYPE.STRING,
            FIELD_TYPE.VAR_STRING,
            FIELD_TYPE.VARCHAR,
            FIELD_TYPE.TINY_BLOB,
            FIELD_TYPE.BLOB,
            FIELD_TYPE.MEDIUM_BLOB,
            FIELD_TYPE.LONG_BLOB,
        ):
            self.assertEqualDecode(b"binary", b"binary", field_type, is_binary=True)

        self.log_ended(test)

    def decode_integer(self) -> None:
        test = "DECODE INTEGER"
        self.log_start(test)

        # INTEGER types
        for value in (
            b"0",
            b"1",
            b"-1",
            b"2345",
            b"-2345",
            b"456789",
            b"-456789",
            b"1234567890",
            b"-1234567890",
            b"9223372036854775807",
            b"-9223372036854775807",
            b"18446744073709551615",
        ):
            expt = int(value)
            for field_type in (
                FIELD_TYPE.TINY,
                FIELD_TYPE.SHORT,
                FIELD_TYPE.LONG,
                FIELD_TYPE.INT24,
                FIELD_TYPE.LONGLONG,
                FIELD_TYPE.YEAR,
            ):
                self.assertEqualDecode(expt, value, field_type)

        # Invalid
        for value in (b"-a1", b"-1a", b"1a", b"a1"):
            for field_type in (
                FIELD_TYPE.TINY,
                FIELD_TYPE.SHORT,
                FIELD_TYPE.LONG,
                FIELD_TYPE.INT24,
                FIELD_TYPE.LONGLONG,
                FIELD_TYPE.YEAR,
            ):
                with self.assertRaises(ValueError):
                    decode(value, field_type, b"utf8", False)

        self.log_ended(test)

    def decode_float(self) -> None:
        test = "DECODE FLOAT"
        self.log_start(test)

        # FLOAT types
        for value in (
            b"0.0",
            b".1",
            b"-.1",
            b"1.",
            b"-1.",
            b"1.1",
            b"-1.1",
            b"2345.67",
            b"-2345.67",
            b"456789.1234",
            b"-456789.1234",
            b"1234567890.5678",
            b"-1234567890.5678",
            b"922337203.6854775807",
            b"-922337203.6854775807",
        ):
            expt = float(value)
            for field_type in (
                FIELD_TYPE.FLOAT,
                FIELD_TYPE.DOUBLE,
            ):
                self.assertEqualDecode(expt, value, field_type)

        # Invalid
        for value in (
            b"-a1.1",
            b"-1a.1",
            b"-1.a1",
            b"-1.1a",
            b"a1.1",
            b"1a.1",
            b"1.a1",
            b"1.1a",
        ):
            for field_type in (
                FIELD_TYPE.TINY,
                FIELD_TYPE.SHORT,
                FIELD_TYPE.LONG,
                FIELD_TYPE.INT24,
                FIELD_TYPE.LONGLONG,
                FIELD_TYPE.YEAR,
            ):
                with self.assertRaises(ValueError):
                    decode(value, field_type, b"utf8", False)

        self.log_ended(test)

    def decode_decimal(self) -> None:
        test = "DECODE DECIMAL"
        self.log_start(test)

        # DECIMAL types
        for value in (
            b"0.0",
            b".1",
            b"-.1",
            b"1.",
            b"-1.",
            b"1.1",
            b"-1.1",
            b"2345.67",
            b"-2345.67",
            b"456789.1234",
            b"-456789.1234",
            b"1234567890.5678",
            b"-1234567890.5678",
            b"922337203.6854775807",
            b"-922337203.6854775807",
        ):
            expt = decimal.Decimal(value.decode("utf8"))
            for field_type in (
                FIELD_TYPE.DECIMAL,
                FIELD_TYPE.NEWDECIMAL,
            ):
                self.assertEqualDecode(expt, value, field_type, use_decimal=True)

        self.log_ended(test)

    def decode_bit(self) -> None:
        test = "DECODE BIT"
        self.log_start(test)

        # BIT type: raw
        for value in (
            b"\x01",
            b"\x01\x01",
            b"\x01\x01\x01",
            b"\x01\x01\x01\x01",
            b"\x01\x01\x01\x01\x01",
            b"\x01\x01\x01\x01\x01\x01",
            b"\x01\x01\x01\x01\x01\x01\x01",
            b"\x01\x01\x01\x01\x01\x01\x01\x01",
            b"\x01\x01\x01\x01\x01\x01\x01\x01\x01",
        ):
            self.assertEqualDecode(value, value, FIELD_TYPE.BIT, decode_bit=False)

        # BIT type: int
        for value, expt in (
            (b"\x01", 1),
            (b"\x01\x01", 257),
            (b"\x01\x01\x01", 65793),
            (b"\x01\x01\x01\x01", 16843009),
            (b"\x01\x01\x01\x01\x01", 4311810305),
            (b"\x01\x01\x01\x01\x01\x01", 1103823438081),
            (b"\x01\x01\x01\x01\x01\x01\x01", 282578800148737),
            (b"\x01\x01\x01\x01\x01\x01\x01\x01", 72340172838076673),
            (b"\x01\x01\x01\x01\x01\x01\x01\x01\x01", 72340172838076673),
        ):
            self.assertEqualDecode(expt, value, FIELD_TYPE.BIT, decode_bit=True)

        self.log_ended(test)

    def decode_date(self) -> None:
        test = "DECODE DATE"
        self.log_start(test)

        # Valid DATE values
        for value in (b"1870-01-01", b"1970-05-05", b"2070-12-31"):
            expt = datetime.date.fromisoformat(value.decode("utf8"))
            for field_type in (
                FIELD_TYPE.DATE,
                FIELD_TYPE.NEWDATE,
            ):
                self.assertEqualDecode(expt, value, field_type)

        # Flexible DATE values
        expt = datetime.date(1, 1, 1)
        for year in (b"1", b"01", b"001", b"0001"):
            for month in (b"1", b"01"):
                for day in (b"1", b"01"):
                    value = year + b"-" + month + b"-" + day
                    for field_type in (
                        FIELD_TYPE.DATE,
                        FIELD_TYPE.NEWDATE,
                    ):
                        self.assertEqualDecode(expt, value, field_type)

        # Invalid DATE values
        for value in (
            b"0000",
            b"0000-12-31",
            b"0001-00-31",
            b"0001-12-00",
        ):
            for field_type in (
                FIELD_TYPE.DATE,
                FIELD_TYPE.NEWDATE,
            ):
                self.assertIsInstance(decode(value, field_type, b"utf8", False), str)

        self.log_ended(test)

    def decode_datetime(self) -> None:
        test = "DECODE DATETIME"
        self.log_start(test)

        # Valid DATETIME values
        for base_value in (
            b"1870-01-01 00:00:00",
            b"1970-05-05 12:34:56",
            b"2070-12-31 23:59:59",
        ):
            for frac in (b"", b".1", b".12", b".123", b".1234", b".12345", b".123456"):
                value = base_value + frac
                expt = datetime.datetime.fromisoformat(value.decode("utf8"))
                for sep in (b" ", b"T"):
                    value = value.replace(b" ", sep)
                    for field_type in (
                        FIELD_TYPE.DATETIME,
                        FIELD_TYPE.TIMESTAMP,
                    ):
                        self.assertEqualDecode(expt, value, field_type)

        # Flexible DATETIME values
        expt = datetime.datetime(1, 1, 1, 1, 1, 1)
        for year in (b"1", b"01", b"001", b"0001"):
            for month in (b"1", b"01"):
                for day in (b"1", b"01"):
                    for hour in (b"1", b"01"):
                        for minute in (b"1", b"01"):
                            for second in (b"1", b"01"):
                                # fmt: off
                                value = year + b"-" + month + b"-" + day + b" " + hour + b":" + minute + b":" + second
                                # fmt: on
                                for field_type in (
                                    FIELD_TYPE.DATETIME,
                                    FIELD_TYPE.TIMESTAMP,
                                ):
                                    self.assertEqualDecode(expt, value, field_type)

        # Invalid DATETIME values
        for value in (
            b"0000",
            b"0000-12-31 00:00:00",
            b"0001-00-31 00:00:00",
            b"0001-12-00 00:00:00",
            b"0001-12-00 24:00:00",
            b"0001-12-00 00:60:00",
            b"0001-12-00 00:00:60",
        ):
            for field_type in (
                FIELD_TYPE.DATETIME,
                FIELD_TYPE.TIMESTAMP,
            ):
                self.assertIsInstance(decode(value, field_type, b"utf8", False), str)

        self.log_ended(test)

    def decode_time(self) -> None:
        test = "DECODE TIME"
        self.log_start(test)

        # Valid TIME values
        for hour in (-25, -24, -23, -1, 0, 1, 23, 24, 25):
            for minute in (-61, -60, -59, -1, 0, 1, 59, 60, 61):
                for second in (-61, -60, -59, -1, 0, 1, 59, 60, 61):
                    for microsecond in (0, 1, 12, 123, 1234, 12345, 123456):
                        td = datetime.timedelta(
                            hours=hour,
                            minutes=minute,
                            seconds=second,
                            microseconds=microsecond,
                        )
                        escaped = escape(td)
                        binary = escaped.encode("utf8").replace(b"'", b"")
                        for type_field in (FIELD_TYPE.TIME,):
                            self.assertEqualDecode(td, binary, type_field)

        # Invalid TIME values
        for value in (
            b"00:1",
            b"a00:00:00.000000",
        ):
            for field_type in (FIELD_TYPE.TIME,):
                self.assertIsInstance(decode(value, field_type, b"utf8", False), str)

        self.log_ended(test)

    def decode_enum(self) -> None:
        test = "DECODE ENUM"
        self.log_start(test)

        # ENUM values
        for value in (b"red", b"green", b"blue"):
            expt = value.decode("utf8")
            for field_type in (FIELD_TYPE.ENUM,):
                self.assertEqualDecode(expt, value, field_type)

        self.log_ended(test)

    def decode_set(self) -> None:
        test = "DECODE SET"
        self.log_start(test)

        # SET values
        for value, expt in (
            (b"red", {"red"}),
            (b"green", {"green"}),
            (b"blue", {"blue"}),
            (b"red,green", {"red", "green"}),
            (b"red,blue", {"red", "blue"}),
            (b"green,blue", {"green", "blue"}),
            (b"red,green,blue", {"red", "green", "blue"}),
        ):
            for field_type in (FIELD_TYPE.SET,):
                self.assertEqualDecode(expt, value, field_type)

        self.log_ended(test)

    def decode_json(self) -> None:
        test = "DECODE JSON"
        self.log_start(test)

        # JSON values
        for value, expt in (
            (b'{"key": "value", "num": 123}', {"key": "value", "num": 123}),
            (b'["a", "b", "c"]', ["a", "b", "c"]),
            (b'{"key": "\xe4\xb8\xad\xe5\x9b\xbd"}', {"key": "中国"}),
            (b'{"key": "Espa\xc3\xb1ol"}', {"key": "Español"}),
        ):
            expt_str = value.decode("utf8")
            for field_type in (FIELD_TYPE.JSON,):
                self.assertEqualDecode(expt, value, field_type, decode_json=True)
                self.assertEqualDecode(expt_str, value, field_type, decode_json=False)

        self.log_ended(test)

    # Utils
    def assertEqualDecode(
        self,
        expt: object,
        data: bytes,
        field_type: int,
        encoding: bytes = b"utf8",
        is_binary: bool = False,
        use_decimal: bool = False,
        decode_bit: bool = False,
        decode_json: bool = False,
    ) -> None:
        res = decode(
            data,
            field_type,
            encoding,
            is_binary,
            use_decimal=use_decimal,
            decode_bit=decode_bit,
            decode_json=decode_json,
        )
        self.assertEqual(type(expt), type(res))
        self.assertEqual(expt, res, f"decode({data!r}) >>> {res!r} != {expt!r}")


if __name__ == "__main__":
    for testcase in (
        TestEscape,
        TestDecode,
    ):
        test = testcase()
        test.test_all()
