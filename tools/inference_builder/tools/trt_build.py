# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import tensorrt as trt

def build_engine(onnx_file_path, min_batch_size, opt_batch_size, max_batch_size, max_text_len):
    # Create a TensorRT logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Read the ONNX file
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    # Set builder configurations
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB of workspace
    config.set_flag(trt.BuilderFlag.FP16)
#    builder.max_batch_size = max_batch_size
    profile = builder.create_optimization_profile()
    for model_input in inputs:
        print(f"Input {model_input.name} with shape {model_input.shape} and dtype {model_input.dtype}")
        input_shape = model_input.shape
        input_name = model_input.name
        if input_name == 'inputs':
            real_shape_min = (min_batch_size, *input_shape[1:])
            real_shape_opt = (opt_batch_size, *input_shape[1:])
            real_shape_max = (max_batch_size, *input_shape[1:])
        elif input_name == 'text_token_mask':
            real_shape_min = (min_batch_size, max_text_len, max_text_len)
            real_shape_opt = (opt_batch_size, max_text_len, max_text_len)
            real_shape_max = (max_batch_size, max_text_len, max_text_len)
        else:
            real_shape_min = (min_batch_size, max_text_len)
            real_shape_opt = (opt_batch_size, max_text_len)
            real_shape_max = (max_batch_size, max_text_len)

        profile.set_shape(input=input_name,
                                min=real_shape_min,
                                opt=real_shape_opt,
                                max=real_shape_max)

    config.add_optimization_profile(profile)

    for output in outputs:
        print(f"Output {output.name} with shape {output.shape} and dtype {output.dtype}")

    # Build and return the engine
    print("Building the engine. This might take a while...")
    engine = builder.build_serialized_network(network, config)
    if engine:
        print("Engine built successfully!")
    return engine

def save_engine(engine, file_name):
    with open(file_name, "wb") as f:
        f.write(engine)

def main(args):
    onnx_file_path = args.input  # Your ONNX file path
    engine_file_path = args.output  # Output TensorRT engine file

    engine = build_engine(onnx_file_path, args.min_batch_size, args.opt_batch_size, args.max_batch_size, args.max_text_len)
    if engine:
        save_engine(engine, engine_file_path)
        print(f"Engine saved to {engine_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRT Builder")
    parser.add_argument("input", type=str, help="Onnx input file")
    parser.add_argument("-o", "--output", type=str, help="TensorRT engine file", default="model.plan")
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=1,
                        help="Maximum batch size for input images")
    parser.add_argument('--min-batch-size',
                        type=int,
                        default=1,
                        help="Minimum batch size for input images")
    parser.add_argument('--opt-batch-size',
                        type=int,
                        default=1,
                        help="Optimal batch size for input images")
    parser.add_argument('--max-text-len',
                        type=int,
                        default=256,
                        help="Maximum text length for input prompt")
    main(parser.parse_args())
