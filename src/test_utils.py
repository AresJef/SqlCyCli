from sqlcycli import utils

if __name__ == "__main__":
    utils._test_find_null_byte()
    utils._test_pack_I24B()
    utils._test_pack_IB()
    utils._test_pack_IIB23s()
    utils._test_pack_unpack_i8()
    utils._test_pack_unpack_i16()
    utils._test_pack_unpack_i24()
    utils._test_pack_unpack_i32()
    utils._test_pack_unpack_i64()
    utils._test_gen_length_encoded_integer()
    utils._test_validate_max_allowed_packet()
