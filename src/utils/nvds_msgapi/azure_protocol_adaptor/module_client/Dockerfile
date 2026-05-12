#######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
#######################################################################################################

# This is the dockerfile to create a container to run the test application to send messages to azure iot edge

# x86 deepstream-<ds-version> image
FROM nvcr.io/nvidia/deepstream:<ds-version>-<tag>

# Jetson deepstream-<ds-version> image
#FROM nvcr.io/nvidia/deepstream-l4t:<ds-version>-<tag>

# copy the compiled sample test applications
COPY test_azure_iotedge_sync /root
COPY test_azure_iotedge_async /root

# copy the azure cfg file
COPY cfg_azure.txt /root
