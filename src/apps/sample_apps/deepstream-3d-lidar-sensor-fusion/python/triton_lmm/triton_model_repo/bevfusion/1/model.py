# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
from typing import Optional
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('/python')

import numpy as np
import torch
import os.path as osp
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union


import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
from torch.utils.dlpack import to_dlpack

from triton_lmm.models.bevfusion.gpu_bevfusion_model import BevFusionModel
from triton_lmm.common.model import INPUT_IMAGE, INPUT_LIDAR, OUTPUT_3D_BBOX, IModel

def load_bevfusion_calibration_data():
    camera_intrinsics = np.array(
        [
            [
                [
                    [1.2664172e+03, 0, 8.1626703e+02, 0], [0, 1.2664172e+03, 4.9150708e+02, 0], [0, 0, 1.0, 0],
                    [0, 0, 0, 1.0]
                ],
                [
                    [1.2608474e+03, 0, 8.0796826e+02, 0], [0, 1.2608474e+03, 4.9533441e+02, 0], [0, 0, 1.0, 0],
                    [0, 0, 0, 1.0]
                ],
                [
                    [1.2725979e+03, 0, 8.2661548e+02, 0], [0, 1.2725979e+03, 4.7975165e+02, 0], [0, 0, 1.0, 0],
                    [0, 0, 0, 1.0]
                ],
                [
                    [8.0922101e+02, 0, 8.2921960e+02, 0], [0, 8.0922101e+02, 4.8177841e+02, 0], [0, 0, 1.0, 0],
                    [0, 0, 0, 1.0]
                ],
                [
                    [1.2567415e+03, 0, 7.9211255e+02, 0], [0, 1.2567415e+03, 4.9277576e+02, 0], [0, 0, 1.0, 0],
                    [0, 0, 0, 1.0]
                ],
                [
                    [1.2595138e+03, 0, 8.0725293e+02, 0], [0, 1.2595138e+03, 5.0119580e+02, 0], [0, 0, 1.0, 0],
                    [0, 0, 0, 1.0]
                ],
            ]
        ], dtype=np.float32)
    assert camera_intrinsics.shape == (1, 6, 4, 4)

    camera2lidar = np.array(
        [
            [
                [
                    [0.999971, 0.00670508, -0.00361619, -0.01262742], [0.00349105, 0.01858322, 0.99982125, 0.76488155],
                    [0.00677108, -0.99980485, 0.01855927, -0.31093138], [0, 0, 0, 1]
                ],
                [
                    [0.55165285, -0.01049361, 0.8340078, 0.4965345], [-0.83366805, 0.02424959, 0.55173326, 0.61423177],
                    [-0.02601402, -0.99965084, 0.00462917, -0.3267865], [0, 0, 0, 1]
                ],
                [
                    [0.57293636, 0.00265955, -0.8195955, -0.49169856], [0.81932807, 0.02389283, 0.5728269, 0.5890876],
                    [0.02110592, -0.999711, 0.01151002, -0.31962872], [0, 0, 0, 1]
                ],
                [
                    [-0.9999412, 0.00981312, -0.00462126, -0.00381823], [
                        0.00469454, 0.00745826, -0.99996114, -0.9087652
                    ], [-0.00977827, -0.99992406, -0.00750389, -0.28325802], [0, 0, 0, 1]
                ],
                [
                    [-0.31705007, 0.0198902, -0.9482002, -0.4831373], [0.94808286, 0.03285569, -0.31632164, 0.09905002],
                    [0.02486207, -0.99926215, -0.02927446, -0.24979047], [0, 0, 0, 1]
                ],
                [
                    [-0.3569066, -0.00545405, 0.9341242, 0.48225552], [-0.9334642, 0.0401144, -0.35642022, 0.07683985],
                    [-0.03552789, -0.9991802, -0.01940826, -0.2732292], [0, 0, 0, 1]
                ]
            ]
        ], dtype=np.float32)
    assert camera2lidar.shape == (1, 6, 4, 4)

    lidar2image = np.array(
        [
            [
                [
                    [1.2634287e+03, 8.2054224e+02, 2.3724329e+01, -6.0428711e+02],
                    [6.7140379e+00, 5.1495331e+02, -1.2570480e+03, -7.8464923e+02],
                    [-3.6161933e-03, 9.9982125e-01, 1.8559270e-02, -7.5901979e-01], [0, 0, 0, 1.0]
                ],
                [
                    [1.3694019e+03, -6.0534521e+02, -2.9059496e+01, -3.1762927e+02],
                    [3.9988193e+02, 3.0386749e+02, -1.2581143e+03, -7.9633502e+02],
                    [8.3400780e-01, 5.5173326e-01, 4.6291691e-03, -7.5149298e-01], [0, 0, 0, 1.0]
                ],
                [
                    [5.1627274e+01, 1.5161827e+03, 3.6373707e+01, -8.5615338e+02],
                    [-3.8981775e+02, 3.0522061e+02, -1.2667081e+03, -7.7635083e+02],
                    [-8.1959552e-01, 5.7282692e-01, 1.1510021e-02, -7.3676026e-01], [0, 0, 0, 1.0]
                ],
                [
                    [-8.1300543e+02, -8.2538843e+02, -1.4135155e+01, -7.5719244e+02],
                    [5.7145619e+00, -4.7572430e+02, -8.1277478e+02, -6.6252484e+02],
                    [-4.6212552e-03, -9.9996114e-01, -7.5038881e-03, -9.1087306e-01], [0, 0, 0, 1.0]
                ],
                [
                    [-1.1495312e+03, 9.4093268e+02, 8.0565224e+00, -6.4656842e+02],
                    [-4.4225323e+02, -1.1458453e+02, -1.2702400e+03, -5.1961328e+02],
                    [-9.4820023e-01, -3.1632164e-01, -2.9274460e-02, -4.3409172e-01], [0, 0, 0, 1.0]
                ],
                [
                    [3.0454568e+02, -1.4634324e+03, -6.0415245e+01, -5.0926125e+01],
                    [4.6130966e+02, -1.2811168e+02, -1.2682085e+03, -5.5913666e+02],
                    [9.3412417e-01, -3.5642022e-01, -1.9408258e-02, -4.2840216e-01], [0, 0, 0, 1.0]
                ]
            ]
        ], dtype=np.float32)
    assert lidar2image.shape == (1, 6, 4, 4)

    img_aug_matrix = np.array(
        [
            [
                [[0.48, 0, 0, -32], [0, 0.48, 0, -176], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0.48, 0, 0, -32], [0, 0.48, 0, -176], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0.48, 0, 0, -32], [0, 0.48, 0, -176], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0.48, 0, 0, -32], [0, 0.48, 0, -176], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0.48, 0, 0, -32], [0, 0.48, 0, -176], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0.48, 0, 0, -32], [0, 0.48, 0, -176], [0, 0, 1, 0], [0, 0, 0, 1]]
            ]
        ], dtype=np.float32)
    assert img_aug_matrix.shape == (1, 6, 4, 4)
    return {
        "camera_intrinsics": camera_intrinsics,
        "camera2lidar": camera2lidar,
        "lidar2image": lidar2image,
        "img_aug_matrix": img_aug_matrix
    }


class TritonPythonModel:

    def initialize(self, args):
        self.end_times = []
        self.inference_times = []

        # export BEVFUSION_MODEL=bevfusion/model/resnet50int8
        # export BEVFUSION_PRECISION=int8
        model_dir = os.environ.get("BEVFUSION_MODEL", "bevfusion/model_root/model/resnet50int8")
        precision = os.environ.get("BEVFUSION_PRECISION", "int8")
        bevfusion = BevFusionModel(model_name="bevfusion", model_dir=model_dir, precision=precision)
        calibration_params = load_bevfusion_calibration_data()
        bevfusion.start(**calibration_params)

        self.inferencer = bevfusion

    def execute(self, requests):
        responses = []
        logger = pb_utils.Logger
        torch.set_default_device('cuda')
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:

            input_lidar = pb_utils.get_input_tensor_by_name(request, 'input_lidar')
            input_lidar_tensor = from_dlpack(input_lidar.to_dlpack())

            input_dict = {'input_lidar': input_lidar_tensor}

            for img_name in self.inferencer._image_names:
                pb_tensor = pb_utils.get_input_tensor_by_name(request, img_name)
                input_dict[img_name] = from_dlpack(pb_tensor.to_dlpack())
            ret = self.inferencer._infer(input_dict)
            out_tensor = pb_utils.Tensor('output_3d_bbox', ret['output_3d_bbox'].astype(np.float32))

            inference_response = pb_utils.InferenceResponse(
                output_tensors = [out_tensor]
            )
            responses.append(inference_response)
        return responses


    def finalize(self):
        print("Cleaning up...")
