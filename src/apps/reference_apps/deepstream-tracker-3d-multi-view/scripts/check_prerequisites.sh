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

# Global variables - same as setup script
BASE_DIR=${BASE_DIR:-$HOME}
USE_INFERENCE_BUILDER=${USE_INFERENCE_BUILDER:-false}
MV3DT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS_DIR="${DS_DIR:-$(git -C "$MV3DT_DIR" rev-parse --show-toplevel)}"
cd "$MV3DT_DIR" || exit 1
KAFKA_VERSION="4.2.0"
SCALA_VERSION="2.13"

# Standardized paths
KAFKA_DIR="$BASE_DIR/kafka_${SCALA_VERSION}-${KAFKA_VERSION}"
INFERENCE_BUILDER_DIR="${INFERENCE_BUILDER_DIR:-$DS_DIR/tools/inference_builder}"

# Initialize status variables
GPU_STATUS="✓"
MQTT_STATUS="✓"
KAFKA_STATUS="✓"
MV3DT_VENV_STATUS="✓"
INFERENCE_BUILDER_STATUS="✓"
DATASETS_MODELS_STATUS="✓"

# Check 1: Check if NVIDIA GPU is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA GPU is not available"
    GPU_STATUS="✗"
fi

# Check 2: Check if MQTT broker is running on port 1883
if ! ./scripts/mosquitto_test.sh > /dev/null 2>&1; then
    MQTT_STATUS="✗"
fi

# Check 3: Check if Kafka is running
if [[ ! -d "$KAFKA_DIR" ]]; then
    echo "Kafka directory not found: $KAFKA_DIR"
    KAFKA_STATUS="✗"
else
    KAFKA_TOPICS_SCRIPT="$KAFKA_DIR/bin/kafka-topics.sh"
    if [[ -x "$KAFKA_TOPICS_SCRIPT" ]]; then
        TOPIC_LIST=$(timeout 5s $KAFKA_TOPICS_SCRIPT --bootstrap-server localhost:9092 --list 2>/dev/null)
        if ! echo "$TOPIC_LIST" | grep -q "mv3dt"; then
            echo "Kafka topics: $TOPIC_LIST"
            echo "mv3dt topic not found"
            KAFKA_STATUS="✗"
        fi
    else
        echo "Kafka topics script not found or not executable: $KAFKA_TOPICS_SCRIPT"
        KAFKA_STATUS="✗"
    fi
fi

# Check 4: Check whether virtual env "mv3dt_venv" setup correctly
if [[ ! -d "$MV3DT_DIR/mv3dt_venv" ]]; then
    echo "Virtual environment 'mv3dt_venv' not found"
    MV3DT_VENV_STATUS="✗"
fi

# Check 5: Check whether inference builder is setup correctly
if [[ "$USE_INFERENCE_BUILDER" != "true" ]]; then
    INFERENCE_BUILDER_STATUS="(skipped)"
else
    if [[ ! -d "$INFERENCE_BUILDER_DIR" ]]; then
        echo "Inference builder directory not found: $INFERENCE_BUILDER_DIR"
        INFERENCE_BUILDER_STATUS="✗"
    elif [[ ! -d "$INFERENCE_BUILDER_DIR/.venv" ]]; then
        echo "Inference builder virtual environment '.venv' not found: $INFERENCE_BUILDER_DIR/.venv"
        INFERENCE_BUILDER_STATUS="✗"
    fi
fi

# Check 6: check wether datasets and models are setup correctly
if [[ ! -d "$MV3DT_DIR/datasets" ]]; then
    echo "Datasets directory not found"
    DATASETS_MODELS_STATUS="✗"
else
    for dataset in mtmc_4cam mtmc_12cam scout wildtrack; do
        for folder in camInfo videos; do
            if [[ ! -d "$MV3DT_DIR/datasets/$dataset/$folder" ]]; then
                echo "datasets/$dataset/$folder directory not found"
                DATASETS_MODELS_STATUS="✗"
            fi
        done
    done
fi

if [[ ! -d "$MV3DT_DIR/models" ]]; then
    echo "Models directory not found"
    DATASETS_MODELS_STATUS="✗"
else
    for model in \
        BodyPose3DNet/bodypose3dnet_accuracy.onnx \
        ClipReID/reid_model.onnx \
        PeopleNetTransformer/peoplenet_transformer_model_op17.onnx \
        PeopleNetTransformer/detector_labels.txt \
        PeopleNetTransformer/custom_parser/libnvds_infercustomparser_tao.so \
        RTDETR/rtdetr_warehouse_v1.0.fp16.onnx \
        RTDETR/detector_labels.txt \
        RTDETR/custom_parser/libnvds_infercustomparser_tao.so \
        PeopleNet2.6.3/resnet34_peoplenet.onnx \
        PeopleNet2.6.3/detector_labels.txt; do
        if [[ ! -s "$MV3DT_DIR/models/$model" ]]; then
            echo "Model asset missing or empty: models/$model"
            DATASETS_MODELS_STATUS="✗"
        fi
    done
fi

# Summary of all checks
echo ""
echo "---- PREREQUISITES CHECK SUMMARY ----"
echo "1. NVIDIA GPU:           $GPU_STATUS"
echo "2. MQTT Broker:          $MQTT_STATUS"
echo "3. Kafka Broker:         $KAFKA_STATUS"
echo "4. Python Environment:   $MV3DT_VENV_STATUS"
echo "5. Datasets & Models:    $DATASETS_MODELS_STATUS"
if [[ "$USE_INFERENCE_BUILDER" == "true" ]]; then
    echo "6. Inference Builder:    $INFERENCE_BUILDER_STATUS"
fi
echo "-------------------------------------"

# Check overall status and exit with appropriate code
inference_check_passed=true
if [[ "$USE_INFERENCE_BUILDER" == "true" && "$INFERENCE_BUILDER_STATUS" != "✓" ]]; then
    inference_check_passed=false
fi

if [[ "$GPU_STATUS" = "✓" && "$MQTT_STATUS" = "✓" && "$KAFKA_STATUS" = "✓" && 
      "$MV3DT_VENV_STATUS" = "✓" && "$DATASETS_MODELS_STATUS" = "✓" && 
      "$inference_check_passed" = true ]]; then
    echo ""
    echo "✅ All prerequisites are properly set up!"
    exit 0
else
    echo ""
    echo "❌ Some prerequisites are missing or not properly configured."
    echo "Please run the setup script or follow manual setup instructions."
    exit 1
fi
