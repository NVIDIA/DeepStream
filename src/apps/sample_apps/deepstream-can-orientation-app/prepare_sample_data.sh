#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

NVDS_VERSION=${NVDS_VERSION:-9.1}
sample_tar=/opt/nvidia/deepstream/deepstream/samples/streams/sample_cans_jpg.tbz2

echo "uncompress  ${sample_tar}"
tar -pxvf ${sample_tar} -C .

from_dir=data/sample0
[[ "$(ls -A ${from_dir})" ]] || \
    { echo "${from_dir} is empty, please download test images into the folder.";  exit -1; }

ref_image=data/reference_sample.jpg
[[ -f "${ref_image}" ]] || \
    { echo "${ref_image} is not exist, please download reference image.";  exit -1; }

to_dir=data/sample1
echo "Copy and format original test image files into folder ${to_dir}"
mkdir -p ${to_dir}
sidx=0
for file_name in `ls $from_dir`;
do
    tofile=$(printf "${to_dir}/test_sample_%04d.jpg" $sidx)
    cp -f ${from_dir}/${file_name} ${tofile}
    ((sidx=sidx+1))
done

png_dir=data/sample2
echo "Convert JPG files into PNG into folder ${png_dir}"
mkdir -p ${png_dir}
sidx=0
for file_name in `ls $from_dir`;
do
    png_file=$(printf "${png_dir}/test_sample_%04d.png" $sidx)
    gst-launch-1.0 -e filesrc location="${from_dir}/${file_name}" ! \
        jpegdec ! videoconvert ! "video/x-raw, format=GRAY8" ! pngenc ! filesink location="${png_file}"
    ((sidx=sidx+1))
done

grey_sample=data/test_samples_raw.grey
echo "Generate raw GREY test sample file ${grey_sample}"
gst-launch-1.0 -e multifilesrc location="${to_dir}/test_sample_%04d.jpg" ! \
    jpegdec ! videoconvert ! "video/x-raw, format=GRAY8" ! \
    filesink location=${grey_sample}

rgba_sample=data/test_samples_raw.rgba
echo "Generate raw RGBA test sample file ${rgba_sample}"
gst-launch-1.0 -e multifilesrc location="${to_dir}/test_sample_%04d.jpg" ! \
    jpegdec ! videoconvert ! "video/x-raw, format=RGBA" ! \
    filesink location=${rgba_sample}

grey_ref_sample=data/reference_sample.grey
echo "Generate GREY reference sample file ${grey_ref_sample}"
gst-launch-1.0 -e filesrc location="${ref_image}" ! \
    jpegdec ! videoconvert ! "video/x-raw, format=GRAY8" ! \
    filesink location=${grey_ref_sample}

echo "Data is ready"

# Patch yaml configs and PFS: fix data URIs to point to local data/ dir and lib path to versioned DS install
DATA_DIR=$(pwd)/data
LIB_PATH=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib
for yaml in ds_can_orientation_jpg.yaml ds_can_orientation_raw.yaml ds_can_orientation_basler_cam.yaml; do
    [[ -f "$yaml" ]] || continue
    sed -i \
        -e "s|multifile:///opt/.*/data/|multifile://${DATA_DIR}/|g" \
        -e "s|file:///opt/.*/data/|file://${DATA_DIR}/|g" \
        -e "s|/opt/nvidia/deepstream/deepstream/lib/|${LIB_PATH}/|g" \
        "$yaml"
done
echo "Updated yaml configs with local data paths"

# Patch Basler PFS emulation config: fix ImageFilename to point to local data/sample2
PFS_FILE=basler_cam_emulation_0815-0000.pfs
if [[ -f "$PFS_FILE" ]]; then
    sed -i -e "s|ImageFilename\t.*|ImageFilename\t${DATA_DIR}/sample2|g" "$PFS_FILE"
    echo "Updated $PFS_FILE with local data path"
fi
