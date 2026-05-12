#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import os


def save(tensor, file, verbose=False):
    if not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)

    dtype_map = {
        "float32": 3,
        "float16": 2,
        "int32": 1,
        "int64": 4,
        "uint64": 5,
        "uint32": 6,
        "int8": 7,
        "uint8": 8,
    }
    if str(tensor.dtype) not in dtype_map:
        raise RuntimeError(f"Unsupport dtype {tensor.dtype}")

    if verbose:
        print(f"Save tensor[{tensor.shape}, {tensor.dtype}] to {file}")

    magic_number = 0x33FF1101
    with open(file, "wb") as f:
        head = np.array(
            [magic_number, tensor.ndim, dtype_map[str(tensor.dtype)]], dtype=np.int32
        ).tobytes()
        f.write(head)

        dims = np.array(tensor.shape, dtype=np.int32).tobytes()
        f.write(dims)

        data = tensor.tobytes()
        f.write(data)


dtype_for_integer_mapping = {
    3: np.float32,
    2: np.float16,
    1: np.int32,
    4: np.int64,
    5: np.uint64,
    6: np.uint32,
    7: np.int8,
    8: np.uint8,
}

dtype_size_mapping = {
    np.float32: 4,
    np.float16: 2,
    np.int32: 4,
    np.int64: 8,
    np.uint64: 8,
    np.uint32: 4,
    np.int8: 1,
    np.uint8: 1,
}


def load_from_buffer(buf):
    offset = 0
    magic_number, ndim, dtype_integer = np.frombuffer(buf[:12], dtype=np.int32)
    if dtype_integer not in dtype_for_integer_mapping:
        raise RuntimeError(f"Can not find match dtype for index {dtype_integer}")

    offset += 12
    dtype = dtype_for_integer_mapping[dtype_integer]
    magic_number_std = 0x33FF1101
    assert magic_number == magic_number_std, f"this file is not tensor file"
    dims = np.frombuffer(buf[offset : ndim * 4 + offset], dtype=np.int32)
    offset += ndim * 4
    volumn = np.cumprod(dims)[-1]
    data = np.frombuffer(
        buf[offset : volumn * dtype_size_mapping[dtype] + offset], dtype=dtype
    ).reshape(*dims)
    return data


def load(file):
    with open(file, "rb") as f:
        size = os.path.getsize(file)
        buf = f.read(size)
        return load_from_buffer(buf)
