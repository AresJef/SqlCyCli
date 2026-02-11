# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.dict import PyDict_SetItem as dict_setitem  # type: ignore
from cython.cimports.cpython.dict import PyDict_GetItem as dict_getitem  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_AsString as bytes_to_chars  # type: ignore

# Python imports
from typing import Iterator, Any
from sqlcycli import errors

__all__ = [
    "Charset",
    "Charsets",
    "all_charsets",
    "by_id",
    "by_name",
    "by_collation",
    "by_name_n_collation",
]


# Charset(s) ----------------------------------------------------------------------------------
@cython.cclass
class Charset:
    """Represents a MySQL character set."""

    _id: cython.int
    _name: str
    _collation: str
    _encoding: bytes
    _encoding_ptr: cython.p_const_char
    _is_default: cython.bint
    _hashcode: cython.Py_ssize_t

    def __init__(
        self,
        id: cython.int,
        name: str,
        collation: str,
        is_default: cython.bint = False,
    ) -> None:
        """The MySQL character set

        :param id `<'int'>`: Numeric MySQL charset identifier.
        :param name `<'str'>`: The charset name (e.g., `"utf8mb4"`).
        :param collation `<'str'>`: The collation name (e.g., `"utf8mb4_general_ci"`).
        :param is_default `<'bool'>`: Whether the charset is one of MySQL's defaults. Defaults to `False`.
        """
        self._id = id
        if name is None:
            raise AssertionError("Charset name cannot be 'None'.")
        self._name = name.lower().strip()
        if collation is None:
            raise AssertionError("Charset collation cannot be 'None'.")
        self._collation = collation.lower().strip()
        if self._name in ("utf8mb4", "utf8mb3"):
            self._encoding = b"utf8"
        elif self._name == "latin1":
            self._encoding = b"cp1252"
        elif self._name == "koi8r":
            self._encoding = b"koi8_r"
        elif self._name == "koi8u":
            self._encoding = b"koi8_u"
        else:
            self._encoding = self._name.encode("ascii", "strict")
        self._encoding_ptr = bytes_to_chars(self._encoding)
        self._is_default = is_default
        self._hashcode = -1

    # Property ---------------------------------------------------------------------
    @property
    def id(self) -> int:
        """Numeric MySQL charset identifier `<'int'>`."""
        return self._id

    @property
    def name(self) -> str:
        """The charset name (e.g., `"utf8mb4"`) `<'str'>`."""
        return self._name

    @property
    def collation(self) -> str:
        """The collation name (e.g., `"utf8mb4_general_ci"`) `<'str'>`."""
        return self._collation

    @property
    def is_default(self) -> bool:
        """Whether the charset is one of MySQL's defaults. `<'bool'>`."""
        return self._is_default

    @property
    def encoding(self) -> bytes:
        """The Python encoding of the charset `<'bytes'>`."""
        return self._encoding

    # Methods ----------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_binary(self) -> cython.bint:
        """Check if this charset is the MySQL binary charset (ID == 63). `<'bool'>`."""
        return self._id == 63

    def __repr__(self) -> str:
        return "<Charset(id=%d, name='%s', collation='%s', encoding=%s)>" % (
            self._id,
            self._name,
            self._collation,
            self._encoding,
        )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Charset):
            _o: Charset = o
            return (
                self._id == _o._id
                and self._name == _o._name
                and self._collation == _o._collation
                and self._is_default == _o._is_default
            )
        return NotImplemented

    def __hash__(self) -> int:
        if self._hashcode == -1:
            self._hashcode = id(self)
        return self._hashcode


@cython.cclass
class Charsets:
    """Collection for managing all the MySQL charsets."""

    _by_id: dict[int, Charset]
    _by_name: dict[str, Charset]
    _by_collation: dict[str, Charset]
    _by_name_n_collation: dict[str, Charset]

    def __init__(self) -> None:
        """Collection for managing all the MySQL charsets."""
        self._by_id = {}
        self._by_name = {}
        self._by_collation = {}
        self._by_name_n_collation = {}

    # Add Charset ------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def add(self, charset: Charset) -> cython.bint:
        """Add MySQL charset to the collection.

        :param charset `<'Charset'>`: The charset instance to add.
        """
        if charset is None:
            raise AssertionError("charset cannot be 'None'.")
        self._index_by_id(charset)
        self._index_by_name(charset)
        self._index_by_collation(charset)
        self._index_by_name_n_collation(charset)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _index_by_id(self, charset: Charset) -> cython.bint:
        """(internal) Index MySQL charset by its id.

        :param charset `<'Charset'>`: The charset instance to index.
        """
        dict_setitem(self._by_id, charset._id, charset)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _index_by_name(self, charset: Charset) -> cython.bint:
        """(internal) Index MySQL charset by its name.

        :param charset `<'Charset'>`: The charset instance to index.
        """
        if charset._is_default:
            dict_setitem(self._by_name, charset._name, charset)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _index_by_collation(self, charset: Charset) -> cython.bint:
        """(internal) Index MySQL charset by its collation.

        :param charset `<'Charset'>`: The charset instance to index.
        """
        dict_setitem(self._by_collation, charset._collation, charset)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _index_by_name_n_collation(self, charset: Charset) -> cython.bint:
        """(internal) Index MySQL charset by its name and collation.

        :param charset `<'Charset'>`: The charset instance to index.
        """
        dict_setitem(
            self._by_name_n_collation,
            self._gen_charset_n_collate_key(charset._name, charset._collation),
            charset,
        )
        return True

    @cython.cfunc
    @cython.inline(True)
    def _gen_charset_n_collate_key(self, name: object, collation: object) -> str:
        """(internal) Generate a unique lookup key from name and collation `<'str'>."""
        return "%s:%s" % (name, collation)

    # Access Charset ---------------------------------------------------------------
    @cython.ccall
    def by_id(self, id: object) -> Charset:
        """Retrieve a Charset by its MySQL ID `<'Charset'>`.

        :param id `<'int'>`: The ID of the charset.
        """
        val = dict_getitem(self._by_id, id)
        if val == cython.NULL:
            raise errors.CharsetNotFoundError(
                "<'%s'>\nMySQL charset ID '%d' does not exist."
                % (self.__class__.__name__, id)
            )
        return cython.cast(Charset, val)

    @cython.ccall
    def by_name(self, name: object) -> Charset:
        """Retrieve a default Charset by its name `<'Charset'>`.

        :param name `<'str'>`: The name of the charset.
        """
        if name in ("utf8mb4", "utf8", "utf-8"):
            return _default_utf8mb4

        val = dict_getitem(self._by_name, name)
        if val == cython.NULL:
            raise errors.CharsetNotFoundError(
                "<'%s'>\nMySQL charactor set '%s' does not exist."
                % (self.__class__.__name__, name)
            )
        return cython.cast(Charset, val)

    @cython.ccall
    def by_collation(self, collation: object) -> Charset:
        """Retrieve a Charset by its collation `<'Charset'>`.

        :param collation `<'str'>`: The collation of the charset.
        """
        if collation == "utf8mb4_general_ci":
            return _default_utf8mb4

        val = dict_getitem(self._by_collation, collation)
        if val == cython.NULL:
            raise errors.CharsetNotFoundError(
                "<'%s'>\nMySQL charactor collation '%s' does not exist."
                % (self.__class__.__name__, collation)
            )
        return cython.cast(Charset, val)

    @cython.ccall
    def by_name_n_collation(self, name: object, collation: object) -> Charset:
        """Retrieve a Charset by both its name and collation combination `<'Charset'>`.

        :param name `<'str'>`: The name of the charset.
        :param collation `<'str'>`: The collation of the charset.
        """
        if name in ("utf8mb4", "utf8", "utf-8"):
            if collation == "utf8mb4_general_ci":
                return _default_utf8mb4
            _key: str = self._gen_charset_n_collate_key("utf8mb4", collation)
        else:
            _key: str = self._gen_charset_n_collate_key(name, collation)

        val = dict_getitem(self._by_name_n_collation, _key)
        if val == cython.NULL:
            raise errors.CharsetNotFoundError(
                "<'%s'>\nMySQL charactor set & collation '%s & %s' does not exist."
                % (self.__class__.__name__, name, collation)
            )
        return cython.cast(Charset, val)

    # Special Methods -------------------------------------------------------------
    def __iter__(self) -> Iterator[Charset]:
        return self._by_id.values().__iter__()

    def __repr__(self) -> str:
        return "<Charsets(\n%s\n)>" % (
            "\n".join(str(charset) for charset in self._by_id.values()),
        )


_charsets: Charsets = Charsets()


# Functions -----------------------------------------------------------------------------------
@cython.ccall
def all_charsets() -> Charsets:
    """Retrieve the collection of all the MySQL charsets `<'Charsets'>`."""
    return _charsets


@cython.ccall
def by_id(id: object) -> Charset:
    """Retrieve a Charset by its MySQL ID `<'Charset'>`.

    :param id `<'int'>`: The ID of the charset.
    """
    return _charsets.by_id(id)


@cython.ccall
def by_name(name: object) -> Charset:
    """Retrieve a default Charset by its name `<'Charset'>`.

    :param name `<'str'>`: The name of the charset.
    """
    return _charsets.by_name(name)


@cython.ccall
def by_collation(collation: object) -> Charset:
    """Retrieve a Charset by its collation `<'Charset'>`.

    :param collation `<'str'>`: The collation of the charset.
    """
    return _charsets.by_collation(collation)


@cython.ccall
def by_name_n_collation(name: str | Any, collation: str | Any) -> Charset:
    """Retrieve a Charset by both its name and collation combination `<'Charset'>`.

    :param name `<'str'>`: The name of the charset.
    :param collation `<'str'>`: The collation of the charset.
    """
    return _charsets.by_name_n_collation(name, collation)


# Add charset ---------------------------------------------------------------------------------
from .constants.CHARSET import CHARSETS

_default_utf8mb4: Charset = None
for id, (name, collation, is_default) in enumerate(CHARSETS):
    ch: Charset = Charset(id, name, collation, bool(is_default))
    if name == "utf8mb4" and collation == "utf8mb4_general_ci":
        _default_utf8mb4 = ch
    _charsets.add(ch)
if _default_utf8mb4 is None:
    raise RuntimeError("Default utf8mb4 charset not found in the CHARSETS list.")

del CHARSETS
