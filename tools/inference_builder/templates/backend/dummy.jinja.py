{#
 SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#}

import numpy as np

class DummyBackend(ModelBackend):
    """Python TensorRT Backend from polygraph"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._model_name = model_config["name"]
        self._input_config = [i for i in model_config['input']]
        self._output_config = [o for o in model_config['output']]
        self._params = model_config["parameters"] if "parameters" in model_config else {}
        n_workers = self._params.get("thread_concurrency", 0)
        # enable thread pool for concurrent inference
        self._thread_pool = ThreadPoolExecutor(max_workers=n_workers) if n_workers > 0 else None
        logger.info(f"DummyBackend created for {self._model_name} to generate {[o['name'] for o in self._output_config]}")

    def __call__(self, *args, **kwargs):
        def generate_sync(in_data):
            for _ in in_data:
                yield self._generate_dummy_data()

        def generate_async(in_data):
            return [self._generate_dummy_data() for _ in in_data]

        in_data = list(args) if args else [kwargs]
        if self._thread_pool is not None:
            return self._thread_pool.submit(generate_async, in_data)
        else:
            return generate_sync(in_data)

    def _generate_dummy_data(self):
        dummy_data = {}
        for config in self._output_config:
            name = config['name']
            dims = config['dims']
            data_type = config.get('data_type', 'TYPE_FP32')

            # Generate random data based on type with correct numpy dtype
            if data_type == 'TYPE_BOOL':
                data = np.random.choice([True, False], size=dims).astype(np.bool_)
            elif data_type == 'TYPE_INT8':
                data = np.random.randint(-128, 127, size=dims, dtype=np.int8)
            elif data_type == 'TYPE_INT16':
                data = np.random.randint(-32768, 32767, size=dims, dtype=np.int16)
            elif data_type == 'TYPE_INT32':
                data = np.random.randint(-100, 100, size=dims, dtype=np.int32)
            elif data_type == 'TYPE_INT64':
                data = np.random.randint(-100, 100, size=dims, dtype=np.int64)
            elif data_type == 'TYPE_UINT8':
                data = np.random.randint(0, 255, size=dims, dtype=np.uint8)
            elif data_type == 'TYPE_UINT16':
                data = np.random.randint(0, 65535, size=dims, dtype=np.uint16)
            elif data_type == 'TYPE_UINT32':
                data = np.random.randint(0, 100, size=dims, dtype=np.uint32)
            elif data_type == 'TYPE_UINT64':
                data = np.random.randint(0, 100, size=dims, dtype=np.uint64)
            elif data_type == 'TYPE_FP16':
                data = np.random.randn(*dims).astype(np.float16)
            elif data_type == 'TYPE_FP32':
                data = np.random.randn(*dims).astype(np.float32)
            elif data_type == 'TYPE_FP64':
                data = np.random.randn(*dims).astype(np.float64)
            elif 'TYPE_STRING' in data_type or 'TYPE_BYTES' in data_type:
                # Generate random strings
                flat_size = np.prod(dims) if dims else 1
                data = np.array([''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=10))
                                for _ in range(flat_size)]).reshape(dims) if dims else np.array(['random_str'])
            else:
                # Default to FP32 for unknown types
                data = np.random.randn(*dims).astype(np.float32)

            dummy_data[name] = data
        return dummy_data

    def _valicate_inputs(self, data):
        i_names = [i['name'] for i in self._input_config]
        i_shapes = {i['name']: i['dims'] for i in self._input_config}
        i_types = {i['name']: i['data_type'] for i in self._input_config}
        for name, tensor in data.items():
            if name not in i_names:
                raise ValueError(f"Input {name} not found in the input config")
            if len(tensor.shape) != len(i_shapes[name]):
                raise ValueError(f"Input {name} has invalid shape: {tensor.shape}")
            for i, s in enumerate(tensor.shape):

                if i_shapes[name][i] != -1 and s != i_shapes[name][i]:
                    raise ValueError(f"Input {name} has invalid shape: {tensor.shape}")
            if isinstance(tensor, np.ndarray) and tensor.dtype != np_datatype_mapping[i_types[name]]:
                raise ValueError(f"Input {name} has invalid type: {tensor.dtype}")
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch_datatype_mapping[i_types[name]]:
                raise ValueError(f"Input {name} has invalid type: {tensor.dtype}")
