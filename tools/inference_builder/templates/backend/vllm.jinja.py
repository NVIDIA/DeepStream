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

import asyncio, threading, uuid, queue
from asyncio import run_coroutine_threadsafe
from typing import Callable, Any, Optional, List, Dict
from concurrent.futures import Future
from vllm import LLM, SamplingParams

class VLLMSyncFacade:
    """
    Wrapper around AsyncLLMEngine that provides a synchronous interface
    for submitting inference requests that can be processed concurrently.
    """
    def __init__(self, **engine_kwargs):
        self.loop = asyncio.new_event_loop()
        self.out_q = queue.Queue(maxsize=1000)
        self._ready = threading.Event()

        def runner():
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            try:
                from vllm.engine.async_llm_engine import AsyncEngineArgs
            except ImportError:
                from vllm.engine.arg_utils import AsyncEngineArgs
            asyncio.set_event_loop(self.loop)
            self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
            self._ready.set()
            self.loop.run_forever()

        threading.Thread(target=runner, daemon=True).start()
        self._ready.wait()
        logger.info("VLLMSyncFacade: AsyncLLMEngine initialized and ready")

    def submit(self,
               prompts: List[Any],
               sampling_params: SamplingParams,
               result_filter: Callable[[Any], Dict[str, Any]],
               job_id_prefix: Optional[str] = None) -> Future:
        """
        Submit requests for inference. AsyncEngine processes one prompt at a time,
        but can handle multiple concurrent requests with different request_ids.

        Args:
            prompts: List of prompts (always a list from __call__)
            sampling_params: SamplingParams object for this request
            result_filter: Function that takes a result and returns a dict
            job_id_prefix: Optional prefix for request IDs
        Returns:
            Future that resolves to list of filtered outputs (list of dicts)
        """
        if not callable(result_filter):
            raise ValueError("result_filter must be a callable")

        job_id_prefix = job_id_prefix or f"batch_{uuid.uuid4().hex[:8]}"

        async def _run_single(prompt, request_id):
            """Process a single prompt with unique request_id"""
            final_output = None
            async for out in self.engine.generate(prompt, sampling_params, request_id):
                final_output = out
            # Apply result_filter to each output
            return final_output

        async def _run_all():
            """Submit all prompts concurrently and collect results"""
            # Create tasks for all prompts with unique request_ids
            tasks = []
            for idx, prompt in enumerate(prompts):
                request_id = f"{job_id_prefix}_req{idx}"
                logger.debug(f"VLLMSyncFacade: Creating task for request {request_id}")
                task = _run_single(prompt, request_id)
                tasks.append(task)

            # Run all tasks concurrently and wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Extract outputs from RequestOutput objects and apply filter
            all_outputs = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"VLLMSyncFacade: Request {idx} failed: {result}")
                    # Add empty result for failed request
                    all_outputs.append({})
                elif result is not None:
                    all_outputs.append(result_filter(result))
                else:
                    logger.warning(f"VLLMSyncFacade: Request {idx} returned unexpected result: {result}")

            return all_outputs

        return run_coroutine_threadsafe(_run_all(), self.loop)

class VLLMBackend(ModelBackend):
    """VLLM Backend"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._params = model_config["parameters"] if "parameters" in model_config else {}
        self._inputs = model_config["input"] if "input" in model_config else []
        self._vlm_input_name = next((i["name"] for i in self._inputs if i["data_type"] == "TYPE_CUSTOM_VLM_INPUT"), None)
        self._async_mode = self._params.get("async_mode", False)

        llm_kwargs = {}
        for key in [
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "dtype",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "kv_cache_dtype",
            "pipeline_parallel_size",
            "max_parallel_loading_workers",
            "enforce_eager",
        ]:
            if key in self._params:
                llm_kwargs[key] = self._params.pop(key)

        if self._async_mode:
            self._llm = VLLMSyncFacade(
                model=model_home,
                **llm_kwargs
            )
        else:
            self._llm = LLM(
                model=model_home,
                **llm_kwargs
            )
        logger.info(f"Model {self._model_name} loaded from {model_home}")

    def extract_outputs(self, result: Any) -> Dict[str, Any]:
        """
        Extracts the outputs from the result based on the configured output names
        Args:
            result: The result object from vLLM generation
        Returns:
            Dictionary mapping output names to their values
        """
        outputs = {n: None for n in self._output_names}
        for name in self._output_names:
            if hasattr(result, name):
                outputs[name] = getattr(result, name)
            else:
                logger.warning(f"VLLMBackend: Output {name} not found in {result}")
        return outputs

    def __call__(self, *args, **kwargs):
        def assemble_input(vlm_input_name: str, vlm_input: List[Any], all_input: Dict[str, Any]) -> Any:
            if vlm_input_name in all_input:
                i = all_input.pop(vlm_input_name)
                if isinstance(i, list):
                    vlm_input.extend(i)
                else:
                    vlm_input.append(i)
            else:
                logger.warning(f"VLLMBackend: Input {vlm_input_name} not found in {all_input}")

        def generate_sync(inputs, sampling_params):
            results = self._llm.generate(inputs, sampling_params)
            for result in results:
                yield self.extract_outputs(result)

        # Extract inputs and parameters
        inputs = []
        params = {}
        if args:
            for i in args:
                assemble_input(self._vlm_input_name, inputs, i)
            params = args[0]
        elif kwargs:
            assemble_input(self._vlm_input_name, inputs, kwargs)
            params = kwargs

        # Convert numpy scalars to Python scalars
        for key, value in params.items():
            if isinstance(value, np.ndarray) and not value.shape:
                params[key] = value.item()

        sampling_params = SamplingParams(**params)

        # Use async facade or sync LLM based on mode
        if self._async_mode:
            return self._llm.submit(inputs, sampling_params, self.extract_outputs)
        else:
            return generate_sync(inputs, sampling_params)

