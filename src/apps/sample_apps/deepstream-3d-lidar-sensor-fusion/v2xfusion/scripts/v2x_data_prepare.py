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

import json
import numpy as np
import tensor
import os
import struct
import shutil
import numpy as np
import sys

# pip install python-lzf
import lzf
import struct
from precompute import *


def load_json(path):
    with open(path) as file:
        data = json.load(file)
    return data


def convert_to_homogeneous(matrix):
    homogeneous_matrix = np.pad(matrix, ((0, 1), (0, 1)), mode="constant")
    homogeneous_matrix[3, 3] = 1
    return homogeneous_matrix


def load_pcd(file, convert=True):
    numpy_pcd_type_mappings = [
        (np.dtype("float32"), ("F", 4)),
        (np.dtype("float64"), ("F", 8)),
        (np.dtype("uint8"), ("U", 1)),
        (np.dtype("uint16"), ("U", 2)),
        (np.dtype("uint32"), ("U", 4)),
        (np.dtype("uint64"), ("U", 8)),
        (np.dtype("int16"), ("I", 2)),
        (np.dtype("int32"), ("I", 4)),
        (np.dtype("int64"), ("I", 8)),
    ]
    pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

    meta = dict()
    with open(file, "rb") as f:
        while True:
            line = str(f.readline().strip(), "utf-8")

            if line.startswith("VERSION"):
                meta["version"] = line[8:]
            elif line.startswith("FIELDS"):
                meta["fields"] = line[7:].split()
            elif line.startswith("SIZE"):
                meta["size"] = list(map(int, line[5:].split()))
            elif line.startswith("TYPE"):
                meta["type"] = line[5:].split()
            elif line.startswith("COUNT"):
                meta["count"] = list(map(int, line[6:].split()))
            elif line.startswith("WIDTH"):
                meta["width"] = int(line[6:])
            elif line.startswith("HEIGHT"):
                meta["height"] = int(line[7:])
            elif line.startswith("VIEWPOINT"):
                meta["viewpoint"] = list(map(float, line[10:].split()))
            elif line.startswith("POINTS"):
                meta["points"] = int(line[7:])
            elif line.startswith("DATA"):
                meta["data_type"] = line[5:]
                break
            elif line.startswith("#"):
                continue
            else:
                raise KeyError(f"Unknow line: {line}")

        dtype = np.dtype(
            list(
                zip(
                    meta["fields"],
                    [
                        pcd_type_to_numpy_type[(t, s)]
                        for t, s in zip(meta["type"], meta["size"])
                    ],
                )
            )
        )
        if meta["data_type"] == "ascii":
            data = np.loadtxt(f, dtype=dtype, delimiter=" ")
        elif meta["data_type"] == "binary":
            rowstep = meta["points"] * dtype.itemsize
            buf = f.read(rowstep)
            data = np.frombuffer(buf, dtype=dtype)
        elif meta["data_type"] == "binary_compressed":
            fmt = "II"
            compressed_size, uncompressed_size = struct.unpack(
                fmt, f.read(struct.calcsize(fmt))
            )
            compressed_data = f.read(compressed_size)
            buf = lzf.decompress(compressed_data, uncompressed_size)
            if len(buf) != uncompressed_size:
                raise IOError("Error decompressing data")

            data = np.zeros(meta["width"], dtype=dtype)
            ix = 0
            for dti in range(len(dtype)):
                dt = dtype[dti]
                bytes = dt.itemsize * meta["width"]
                column = np.frombuffer(buf[ix : (ix + bytes)], dt)
                data[dtype.names[dti]] = column
                ix += bytes

        if convert:
            output = np.zeros((len(data), len(dtype)), dtype=np.float32)
            for ic in range(len(dtype)):
                output[:, ic] = data[dtype.names[ic]]
            data = output

    return data


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} V2X-Seq-SPD-Example outputdir")
        exit(1)

    infrastructure_side_data_path = f"{sys.argv[1]}/infrastructure-side/"
    dump_path = f"{sys.argv[2]}"  # The path you want to save

    data = load_json(os.path.join(infrastructure_side_data_path, "data_info.json"))

    dump_count = 200
    keymap = {"0058": "0"}
    dump_info = {"0058": 0}

    base_path = os.path.join(dump_path, str(0))
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "camera"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "lidar"), exist_ok=True)

    # 1. Load data
    for item in data:
        sequence_id = item.get("sequence_id")
        if sequence_id == "0058":
            sequence_dump_path = os.path.join(dump_path, keymap[sequence_id])
            sequence_dump_camera_path = os.path.join(sequence_dump_path, "camera")
            sequence_dump_lidar_path = os.path.join(sequence_dump_path, "lidar")

            if dump_info[sequence_id] == 0:
                camera_intrinsic_path = item.get("calib_camera_intrinsic_path")
                lidar_to_camera_path = item.get("calib_virtuallidar_to_camera_path")

                camera_intrinsic = load_json(
                    os.path.join(infrastructure_side_data_path, camera_intrinsic_path)
                )
                lidar_to_camera = load_json(
                    os.path.join(infrastructure_side_data_path, lidar_to_camera_path)
                )

                camera_intrinsic_matrix = convert_to_homogeneous(
                    np.array(camera_intrinsic["cam_K"]).reshape(3, 3)
                )
                lidar_to_camera_matrix = np.hstack(
                    (
                        np.array(lidar_to_camera["rotation"]),
                        np.array(lidar_to_camera["translation"]),
                    )
                )
                lidar_to_camera_matrix = np.vstack(
                    (lidar_to_camera_matrix, [0, 0, 0, 1])
                )
                camera_to_lidar_matrix = np.linalg.inv(lidar_to_camera_matrix)

                dst_camera_intrinsic_path = os.path.join(
                    sequence_dump_path, "camera_intrinsic.tensor"
                )
                dst_lidar_to_camera_path = os.path.join(
                    sequence_dump_path, "lidar2camera.tensor"
                )
                dst_camera_to_lidar_path = os.path.join(
                    sequence_dump_path, "camera2lidar.tensor"
                )

                tensor.save(
                    camera_intrinsic_matrix.astype(np.float32),
                    dst_camera_intrinsic_path,
                )
                tensor.save(
                    lidar_to_camera_matrix.astype(np.float32), dst_lidar_to_camera_path
                )
                tensor.save(
                    camera_to_lidar_matrix.astype(np.float32), dst_camera_to_lidar_path
                )

            if dump_info[sequence_id] == 200:
                continue

            img_path = os.path.join(infrastructure_side_data_path, item["image_path"])
            lidar_path = os.path.join(
                infrastructure_side_data_path, item["pointcloud_path"]
            )

            index = str(dump_info[sequence_id]).zfill(5)
            # dst lidar path
            dst_lidar_path = os.path.join(
                sequence_dump_lidar_path, "lidar_" + index + ".bin"
            )
            lidar_data = load_pcd(lidar_path)
            with open(dst_lidar_path, "wb") as f:
                f.write(lidar_data.tobytes())
            # lidar_data = pcd_to_bin(lidar_path, out_path = dst_lidar_path)

            # dst img path
            dst_image_path = os.path.join(
                sequence_dump_camera_path, "camera_" + index + ".jpg"
            )
            shutil.copy(img_path, dst_image_path)

            dump_info[sequence_id] += 1

    # 1.1 To demonstrate the case with batch size of 4
    # copy three more copies of the dataset.
    shutil.copytree(base_path, os.path.join(dump_path, str(1)))
    shutil.copytree(base_path, os.path.join(dump_path, str(2)))
    shutil.copytree(base_path, os.path.join(dump_path, str(3)))

    # Image size (1920,1080)  -> (1536, 864)
    img_aug_matrix = np.array(
        [[1536.0 / 1920.0, 0, 0], [0, 864.0 / 1080.0, 0], [0, 0, 1]], dtype=np.float32
    )
    img_aug_matrix = convert_to_homogeneous(img_aug_matrix)

    # 2 merge_matrix
    camera_intrinsics = []
    camera2lidars = []
    lidar2cameras = []
    img_aug_matrics = []
    for i in range(4):
        camera_intrinsics.append(
            tensor.load(f"{dump_path}/{i}/camera_intrinsic.tensor")
        )
        camera2lidars.append(tensor.load(f"{dump_path}/{i}/camera2lidar.tensor"))
        lidar2cameras.append(tensor.load(f"{dump_path}/{i}/lidar2camera.tensor"))
        img_aug_matrics.append(img_aug_matrix)

    camera_intrinsics = np.stack(camera_intrinsics)[:, None]
    camera2lidars = np.stack(camera2lidars)[:, None]
    lidar2cameras = np.stack(lidar2cameras)[:, None]
    img_aug_matrics = np.stack(img_aug_matrics)[:, None]

    # 3 generate intervals&geometrys
    np.set_printoptions(precision=4, suppress=True)
    image_size = [864, 1536]
    downsample_factor = 16
    C = 80  # feature channel
    xbound = [0, 102.4, 0.8]
    ybound = [-51.2, 51.2, 0.8]
    zbound = [-5, 3, 8]
    dbound = [-1.5, 3.0, 180]
    dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)

    denorms = np.zeros((lidar2cameras.shape[0], 4))
    for i in range(lidar2cameras.shape[0]):
        denorms[i] = get_denorm(lidar2cameras[i, 0])

    frustum_rays, rays = create_frustum_rays(image_size, downsample_factor, dbound)
    geom = get_geometry_rays(
        camera2lidars,
        camera_intrinsics,
        img_aug_matrics,
        None,
        denorms,
        frustum_rays,
        rays,
        dbound,
    )
    intervals, geom_feats = pre_compute(geom, C, bx, dx, nx)

    tensor.save(intervals, f"{dump_path}/../intervals.tensor")
    tensor.save(geom_feats, f"{dump_path}/../geometrys.tensor")
