from numpy import int64, uint, uint8
import numpy as np


class Validator:
    name = "output-types-validator"

    def __init__(self, config):
        pass

    def __call__(self, *args):
        int8_outputs = args[0]
        int16_outputs = args[1]
        int32_outputs = args[2]
        int64_outputs = args[3]
        uint8_outputs = args[4]
        uint16_outputs = args[5]
        uint32_outputs = args[6]
        uint64_outputs = args[7]
        fp16_outputs = args[8]
        fp32_outputs = args[9]
        fp64_outputs = args[10]
        bool_outputs = args[11]
        string_outputs = args[12]

        # Type assertions
        assert isinstance(int8_outputs, np.ndarray), f"int8_outputs should be ndarray, got {type(int8_outputs)}"
        assert int8_outputs.dtype == np.int8, f"int8_outputs should have dtype int8, got {int8_outputs.dtype}"

        assert isinstance(int16_outputs, np.ndarray), f"int16_outputs should be ndarray, got {type(int16_outputs)}"
        assert int16_outputs.dtype == np.int16, f"int16_outputs should have dtype int16, got {int16_outputs.dtype}"

        assert isinstance(int32_outputs, np.ndarray), f"int32_outputs should be ndarray, got {type(int32_outputs)}"
        assert int32_outputs.dtype == np.int32, f"int32_outputs should have dtype int32, got {int32_outputs.dtype}"

        assert isinstance(int64_outputs, np.ndarray), f"int64_outputs should be ndarray, got {type(int64_outputs)}"
        assert int64_outputs.dtype == np.int64, f"int64_outputs should have dtype int64, got {int64_outputs.dtype}"

        assert isinstance(uint8_outputs, np.ndarray), f"uint8_outputs should be ndarray, got {type(uint8_outputs)}"
        assert uint8_outputs.dtype == np.uint8, f"uint8_outputs should have dtype uint8, got {uint8_outputs.dtype}"

        assert isinstance(uint16_outputs, np.ndarray), f"uint16_outputs should be ndarray, got {type(uint16_outputs)}"
        assert uint16_outputs.dtype == np.uint16, f"uint16_outputs should have dtype uint16, got {uint16_outputs.dtype}"

        assert isinstance(uint32_outputs, np.ndarray), f"uint32_outputs should be ndarray, got {type(uint32_outputs)}"
        assert uint32_outputs.dtype == np.uint32, f"uint32_outputs should have dtype uint32, got {uint32_outputs.dtype}"

        assert isinstance(uint64_outputs, np.ndarray), f"uint64_outputs should be ndarray, got {type(uint64_outputs)}"
        assert uint64_outputs.dtype == np.uint64, f"uint64_outputs should have dtype uint64, got {uint64_outputs.dtype}"

        assert isinstance(fp16_outputs, np.ndarray), f"fp16_outputs should be ndarray, got {type(fp16_outputs)}"
        assert fp16_outputs.dtype == np.float16, f"fp16_outputs should have dtype float16, got {fp16_outputs.dtype}"

        assert isinstance(fp32_outputs, np.ndarray), f"fp32_outputs should be ndarray, got {type(fp32_outputs)}"
        assert fp32_outputs.dtype == np.float32, f"fp32_outputs should have dtype float32, got {fp32_outputs.dtype}"

        assert isinstance(fp64_outputs, np.ndarray), f"fp64_outputs should be ndarray, got {type(fp64_outputs)}"
        assert fp64_outputs.dtype == np.float64, f"fp64_outputs should have dtype float64, got {fp64_outputs.dtype}"

        assert isinstance(bool_outputs, np.ndarray), f"bool_outputs should be ndarray, got {type(bool_outputs)}"
        assert bool_outputs.dtype == np.bool_, f"bool_outputs should have dtype bool_, got {bool_outputs.dtype}"

        assert isinstance(string_outputs, np.ndarray), f"string_outputs should be ndarray, got {type(string_outputs)}"
        # String arrays can be np.bytes_ (|S*), np.str_ (U*), or np.object_
        assert np.issubdtype(string_outputs.dtype, np.bytes_) or np.issubdtype(string_outputs.dtype, np.str_) or string_outputs.dtype == np.object_, \
            f"string_outputs should have string-like dtype, got {string_outputs.dtype}"

        return [],
