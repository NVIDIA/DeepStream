#!/usr/bin/env bash
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

set -euo pipefail

MV3DT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REPO="$(realpath -m "${MODEL_REPO:-$MV3DT_DIR/models}")"
CACHE_DIR="$MV3DT_DIR/experiments/model-setup"
CLIP_REID_IMAGE="${CLIP_REID_IMAGE:-nvcr.io/nvidia/vss-core/reid-embed:3.3.0}"
NGC_URL=https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao
mkdir -p "$CACHE_DIR" "$MODEL_REPO"

download() {
    local url="$1" destination="$2" checksum="${3:-}"
    local partial="$CACHE_DIR/$(basename "$destination").part"
    if [[ -s "$destination" ]]; then
        if [[ -n "$checksum" ]]; then
            printf '%s  %s\n' "$checksum" "$destination" | sha256sum -c -
        fi
        echo "Using existing $destination"
        return
    fi
    echo "Downloading $destination"
    wget -q --https-only --timeout=60 --tries=3 "$url" -O "$partial"
    test -s "$partial"
    if [[ -n "$checksum" ]]; then
        printf '%s  %s\n' "$checksum" "$partial" | sha256sum -c -
    fi
    mkdir -p "$(dirname "$destination")"
    mv "$partial" "$destination"
}

download "$NGC_URL/peoplenet_transformer_v2/deployable_v1.0/files?redirect=true&path=dino_fan_small_astro_delta.onnx" \
    "$MODEL_REPO/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx"
download "$NGC_URL/rtdetr_2d_warehouse/deployable_efficientvit_l2_v1.0/files?redirect=true&path=rtdetr_warehouse_v1.0.fp16.onnx" \
    "$MODEL_REPO/RTDETR/rtdetr_warehouse_v1.0.fp16.onnx"
download "$NGC_URL/peoplenet/deployable_quantized_onnx_v2.6.3/files?redirect=true&path=resnet34_peoplenet.onnx" \
    "$MODEL_REPO/PeopleNet2.6.3/resnet34_peoplenet.onnx" \
    a6aa31618f8476261807b5ba26b9841c684708c66a5f0be460a25a972f8ee7ad
download "$NGC_URL/bodypose3dnet/deployable_accuracy_onnx_1.0/files?redirect=true&path=bodypose3dnet_accuracy.onnx" \
    "$MODEL_REPO/BodyPose3DNet/bodypose3dnet_accuracy.onnx"

if [[ ! -s "$MODEL_REPO/ClipReID/reid_model.onnx" ]]; then
    # Pin the author source and NVIDIA's checkpoint-verified ONNX converter.
    CLIP_REVISION=eb1898b72c882875f478bebfc6d41644eece0a5d
    VSS_REVISION=3420e4e96c3fcc1b1635938df08969dcf3d43eb0
    EXPORT_DIR="$CACHE_DIR/clip-reid"
    mkdir -p "$EXPORT_DIR/src"
    download "https://codeload.github.com/Syliz517/CLIP-ReID/tar.gz/$CLIP_REVISION" \
        "$EXPORT_DIR/source.tar.gz" \
        7e5ed316fef67344d4ca692739fab07f8170965d1953fc7c3d3d5ec435eb6072
    tar -xzf "$EXPORT_DIR/source.tar.gz" -C "$EXPORT_DIR/src" --strip-components=1
    sed -i 's/if get_text == True:/if get_text:/; s/if get_image == True:/if get_image:/' \
        "$EXPORT_DIR/src/model/make_model_clipreid.py"
    download "https://raw.githubusercontent.com/NVIDIA-AI-Blueprints/video-search-and-summarization/$VSS_REVISION/deploy/docker/services/rtvi/reid-embed/convert_clipreid_to_onnx.py" \
        "$EXPORT_DIR/convert_clipreid_to_onnx.py" \
        a43a639eeb8cf7286af1a3cc7ef664b9e247538e30c2904ea30c48894b79bb3b

    echo "Exporting CLIP ReID with $CLIP_REID_IMAGE (first setup requires GPU access)"
    docker run --rm --gpus all --network bridge \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -v "$EXPORT_DIR:/work" -w /work --entrypoint /bin/bash \
        "$CLIP_REID_IMAGE" -euc '
python3 -m pip install --quiet gdown==5.2.0 yacs==0.1.8 timm==1.0.15 ftfy==6.3.1
if ! echo "6e11721abdc91939da69916c2e109302eb400d0b95196286a18c59d459b79bef  checkpoint.pth" | sha256sum -c - >/dev/null 2>&1; then
    python3 -m gdown 1K32xrosw0gPrxYCWXER81mhWObEW5-d4 -O checkpoint.pth
fi
python3 convert_clipreid_to_onnx.py --repo-dir /work/src --checkpoint /work/checkpoint.pth --output /work/reid_model.onnx
python3 -c "import onnx; model=onnx.load(\"reid_model.onnx\"); onnx.checker.check_model(model); assert model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value == 1280"
'
    test -s "$EXPORT_DIR/reid_model.onnx"
    mkdir -p "$MODEL_REPO/ClipReID"
    mv "$EXPORT_DIR/reid_model.onnx" "$MODEL_REPO/ClipReID/reid_model.onnx"
fi

echo "Models ready: $MODEL_REPO"
