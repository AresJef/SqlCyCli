# cython: language_level=3

# Constants
cdef:
    # orjson helpers
    object _orjson_loads
    object _orjson_dumps
    object _OPT_SERIALIZE_NUMPY

# Custom Escape Types
cdef class ObjStr:
    pass
    
cdef class CustomEscapeType(ObjStr):
    cdef object _obj

cdef class BIT(CustomEscapeType):
    pass

cdef class JSON(CustomEscapeType):
    pass

# Escape
cpdef object escape(object data, bint many=?, bint itemize=?)

# Decode
cpdef object decode(bytes value, unsigned int field_type, const char* encoding, bint is_binary, bint use_decimal=?, bint decode_bit=?, bint decode_json=?)
