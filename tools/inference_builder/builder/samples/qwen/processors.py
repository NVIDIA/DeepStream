# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from transformers import AutoProcessor
import torch
import numpy as np

class QwenVLProcessor:
    """
    Preprocessor which uses qwen_vl_utils to process vision information.
    Works with transformers as demonstrated in the official example:
    https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    """
    name = "qwen-vl-processor"
    def __init__(self, config):
        from qwen_vl_utils import process_vision_info
        model_home = config["model_home"]
        self._processor = AutoProcessor.from_pretrained(model_home)
        self._process_vision_info = process_vision_info

    def __call__(self, *args):
        messages = args[0]
        max_new_tokens = args[1]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        if "pixel_values" in inputs:
            return inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_grid_thw"], None, None, max_new_tokens, inputs["input_ids"]
        else:
            return inputs["input_ids"], inputs["attention_mask"], None, None, inputs["pixel_values_videos"], inputs["video_grid_thw"], max_new_tokens, inputs["input_ids"]

    def apply_chat_template(self, prompt, multimodal_data):
        # Build content list with media placeholders
        content = []

        # Add one placeholder for each media item
        for media_type, items in multimodal_data.items():
            for _ in items:
                content.append({"type": media_type})

        # Add the text content
        content.append({
            "type": "text",
            "text": prompt
        })

        conversation = [{"role": "user", "content": content}]
        return self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

class QwenVLLoader(QwenVLProcessor):
    """
    Preprocessor for loading vllm compatible input for Qwen2.5-VL models.
    """
    name = "qwen-vl-loader"
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, *args):
        messages = args[0]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        mm_data = {}
        if image_inputs:
            mm_data["image"] = image_inputs
        if video_inputs:
            mm_data["video"] = video_inputs
        return {
            "prompt": text,
            "multi_modal_data": mm_data
        }

class Qwen3VLLoader(QwenVLProcessor):
    """
    Preprocessor for loading vllm compatible input for Qwen3-VL models.
    """
    name = "qwen3-vl-loader"
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, *args):
        messages = args[0]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        mm_data = {}
        if image_inputs:
            mm_data["image"] = image_inputs
        if video_inputs and len(video_inputs) > 0:
            fps = 30
            total_num_frames = video_inputs[0].shape[0]
            mm_data["video"] = (video_inputs, {
                "total_num_frames": total_num_frames,
                "frames_indices": list(range(total_num_frames)),
                "fps": fps,
                "duration": total_num_frames/fps,
            })
        mm_processor_kwargs = {
                "do_sample_frames": False,
            }
        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": mm_processor_kwargs,
        }

class QwenVLVideoProcessor(QwenVLProcessor):
    """
    Preprocessor for loading video tensors to TensorRT-LLM
    compatible input for Qwen2.5-VL models.
    """
    name = "qwen-vl-video-processor"
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, *args):
        # Convert from DLPack to torch tensor, then transform from HWC to CHW and normalize
        prompts = args[0]
        videos = args[1]
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        elif not isinstance(prompts, list):
            prompts = [prompts]
            videos = [videos]

        inputs = []
        for prompt, frames in zip(prompts, videos):
            tensors = [
                torch.utils.dlpack.from_dlpack(frame.tensor).permute(2, 0, 1).float() / 255.0
                for frame in frames
            ]
            multimodal_data = {"video": [tensors]}
            inputs.append({
                "prompt": self.apply_chat_template(prompt, multimodal_data),
                "multi_modal_data": multimodal_data
            })
        return inputs


class QwenVLImageProcessor(QwenVLProcessor):
    """
    Preprocessor for loading image tensors to TensorRT-LLM
    compatible input for Qwen2.5-VL models.
    """
    name = "qwen-vl-image-processor"
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, *args):
        # Convert from DLPack to torch tensor, then transform from HWC to CHW and normalize
        prompts = args[0]
        images = args[1]
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        elif not isinstance(prompts, list):
            prompts = [prompts]
            images = [images]

        inputs = []
        for prompt, image in zip(prompts, images):
            tensors = [image.permute(2, 0, 1).float() / 255.0]
            multimodal_data = {"image": tensors}
            inputs.append({
                "prompt": self.apply_chat_template(prompt, multimodal_data),
                "multi_modal_data": multimodal_data
            })
        return inputs

class QwenVLImageCoordinator:
    """
    Preprocessor for loading image tensors to vLLM compatible input
    for Qwen2.5-VL/Qwen3-VL models.
    """
    name = "qwen-vl-image-coordinator"
    def __init__(self, config):
        model_home = config["model_home"]
        self._processor = AutoProcessor.from_pretrained(model_home)

    def __call__(self, *args):
        messages = args[0]
        images = args[1]
        if isinstance(images, np.ndarray):
            images = images.tolist()
        elif not isinstance(images, list):
            images = [images]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        multimodal_data = {"image": [tensor.permute(2, 0, 1).float().cpu() for tensor in images]}
        return {
            "prompt": text,
            "multi_modal_data": multimodal_data
        }

class QwenVLVideoCoordinator:
    """
    Preprocessor for loading video tensors to vLLM compatible input
    for Qwen2.5-VL models.
    """
    name = "qwen-vl-video-coordinator"
    def __init__(self, config):
        model_home = config["model_home"]
        self._processor = AutoProcessor.from_pretrained(model_home)

    def __call__(self, *args):
        messages = args[0]
        videos = args[1]
        if not videos:
            raise ValueError("At least one video is required")
        tensors = [
            torch.utils.dlpack.from_dlpack(frame.tensor).permute(2, 0, 1).float()
            for frame in videos[0] # only one video is supported
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        multimodal_data = {"video": [torch.stack(tensors).cpu()]}
        return {
            "prompt": text,
            "multi_modal_data": multimodal_data
        }

class Qwen3VLVideoCoordinator:
    """
    Preprocessor for loading video tensors to vLLM compatible input
    for Qwen3-VL models.
    """
    name = "qwen3-vl-video-coordinator"
    def __init__(self, config):
        model_home = config["model_home"]
        self._processor = AutoProcessor.from_pretrained(model_home)

    def __call__(self, *args):
        messages = args[0]
        videos = args[1]
        if not videos:
            raise ValueError("At least one video is required")
        tensors = []
        timestamps = []
        # only one video is supported
        for frame in videos[0]:
            tensors.append(torch.utils.dlpack.from_dlpack(frame.tensor).permute(2, 0, 1).float())
            timestamps.append(float(frame.timestamp)/1e9)
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        multimodal_data = {
            "video": ([torch.stack(tensors).cpu()],
            {
                "total_num_frames": len(tensors),
                "frames_indices": list(range(len(tensors))),
                "fps": len(tensors)/(timestamps[-1]-timestamps[0]),
                "duration": timestamps[-1]-timestamps[0],
            })
        }
        return {
            "prompt": text,
            "multi_modal_data": multimodal_data
        }

class QwenVLImageLoader:
    name = "qwen-vl-image-loader"
    def __init__(self, config):
        from tensorrt_llm.inputs import default_multimodal_input_loader
        self._default_image_loader = default_multimodal_input_loader
        self._model_home = config["model_home"]

    def __call__(self, *args):
        prompts = args[0].tolist() if isinstance(args[0], np.ndarray) else args[0]
        images = args[1].tolist() if isinstance(args[1], np.ndarray) else args[1]
        assert len(images) == len(prompts)
        inputs = self._default_image_loader(
            tokenizer=None,
            model_type="qwen2_5_vl",
            model_dir=self._model_home,
            prompts=prompts,
            media=images,
            modality="image"
        )
        return inputs

class QwenVLVideoLoader:
    name = "qwen-vl-video-loader"
    def __init__(self, config):
        from tensorrt_llm.inputs import default_multimodal_input_loader
        self._default_video_loader = default_multimodal_input_loader
        self._model_home = config["model_home"]
        self._num_frames = config.get("num_frames", 8)

    def __call__(self, *args):
        prompts = args[0]
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        elif not isinstance(prompts, list):
            prompts = [prompts]
        videos = args[1]
        if isinstance(videos, np.ndarray):
            videos = videos.tolist()
        elif not isinstance(videos, list):
            videos = [videos]
        assert len(videos) == len(prompts)
        inputs = self._default_video_loader(
            tokenizer=None,
            model_type="qwen2_5_vl",
            model_dir=self._model_home,
            prompts=prompts,
            media=videos,
            modality="video",
            num_frames=self._num_frames
        )
        return inputs

class QwenVLTokenizer:
    name = "qwen-vl-tokenizer"
    def __init__(self, config):
        self._processor = AutoProcessor.from_pretrained(config["model_home"])

    def __call__(self, *args):
        generated_ids = args[0]
        input_ids = args[1]
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids.unsqueeze(0))
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )