# SPDX-FileCopyrightText: Copyright (c) 2017-2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

all:
	gcc -ggdb -c -o dsexample_lib.o -fPIC dsexample_lib.c
	ar rcs libdsexample.a dsexample_lib.o
