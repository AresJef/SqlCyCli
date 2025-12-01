# cython: language_level=3

# Mysql Exceptions
cdef dict MYSQL_ERROR_MAP
cpdef bint raise_mysql_exception(const char* data, Py_ssize_t data_size) except -1