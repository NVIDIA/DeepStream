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

from pyservicemaker import Pipeline, Flow, BufferProvider, Buffer, RenderMode, BufferRetriever, as_tensor, StateTransitionMessage, DynamicSourceMessage, SourceConfig, BatchMetadataOperator, Probe
from typing import Dict, List
from queue import Queue, Empty, Full
from dataclasses import dataclass
import base64
import numpy as np
from abc import ABC, abstractmethod
import yaml
import tempfile
import os

png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGK6HcwNCAAA//8DTgE8HuxwEQAAAABJRU5ErkJggg==")
jpg_data = base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigAooooAKKKKACiiigD/2Q==")


@dataclass
class PerfConfig:
    enable_fps_logs: bool = False
    enable_latency_logs: bool = False

@dataclass
class RenderConfig:
    enable_display: bool = False
    enable_osd: bool = False
    enable_stream: bool = False
    rtsp_mount_point: str | None = "/ds-test"
    rtsp_port: int | None = 8554
    seg_mask_config: dict | None = None

    def __bool__(self) -> bool:
        """Check if render configuration is valid."""
        if self.enable_osd and not self.enable_display and not self.enable_stream:
            logger.warning("RenderConfig: enable_osd is True but both enable_display and enable_stream are False. OSD requires an output method.")
            return False
        if self.enable_stream and (self.rtsp_mount_point is None or self.rtsp_port is None):
            logger.warning("RenderConfig: enable_stream is True but rtsp_mount_point or rtsp_port is None.")
            return False
        return True

@dataclass
class TrackerConfig:
    config_path: str | None = None
    lib_path: str | None = None
    width: int = 1920
    height: int = 1088
    display_tracking_id: bool = False

    def __bool__(self) -> bool:
        """Check if all required tracker configuration fields are set."""
        # Check if width/height is set than we need to have both, set default values if not set
        if (self.width is not None and self.height is None) or (self.width is None and self.height is not None):
            logger.warning("TrackerConfig: width and height must be set together, setting default values")
        return self.config_path is not None and self.lib_path is not None

@dataclass
class MessageBrokerConfig:
    proto_lib_path: str | None = None
    conn_str: str | None = None
    topic: str | None = None
    msgconv_config_path: str | None = None
    msgconv_payload_type: int = 0
    msgconv_msg2p_new_api: bool = False
    msgconv_frame_interval: int = 30
    msgconv_msg2p_lib: str | None = None

    def __bool__(self) -> bool:
        """Check if all required message broker configuration fields are set."""
        if self.msgconv_msg2p_lib is not None and not os.path.exists(self.msgconv_msg2p_lib):
            logger.warning(f"MessageBrokerConfig: msgconv_msg2p_lib does not exist: {self.msgconv_msg2p_lib}")
            return False
        if self.msgconv_config_path is not None and not os.path.exists(self.msgconv_config_path):
            logger.warning(f"MessageBrokerConfig: msgconv_config_path does not exist: {self.msgconv_config_path}")
            return False
        if self.proto_lib_path is not None and not os.path.exists(self.proto_lib_path):
            logger.warning(f"MessageBrokerConfig: proto_lib_path does not exist: {self.proto_lib_path}")
            return False
        return (
            self.proto_lib_path is not None and
            self.msgconv_config_path is not None and
            self.conn_str is not None and
            self.topic is not None
        )

@dataclass
class AnalyticsConfig:
    """Configuration for nvdsanalytics (ROI, line-crossing, overcrowding, direction)."""
    config_path: str | None = None
    gpu_id: int = 0

    def __bool__(self) -> bool:
        """Check if analytics is configured (config_path set)."""
        if self.config_path is None:
            return False
        if not os.path.exists(self.config_path):
            logger.warning(f"AnalyticsConfig: config_path does not exist: {self.config_path}")
            return False
        return True

@dataclass
class KittiConfig:
    infer_kitti_output_dir: str | None = None
    tracker_kitti_output_dir: str | None = None

    def __bool__(self) -> bool:
        """Check if any required kitti configuration fields are set."""
        if self.infer_kitti_output_dir is None and self.tracker_kitti_output_dir is None:
            logger.warning("KittiConfig: No kitti output directory specified")
            return False
        if self.infer_kitti_output_dir is not None and not os.path.exists(self.infer_kitti_output_dir):
            logger.warning(f"KittiConfig: infer_kitti_output_dir does not exist: {self.infer_kitti_output_dir}")
            return False
        if self.tracker_kitti_output_dir is not None and not os.path.exists(self.tracker_kitti_output_dir):
            logger.warning(f"KittiConfig: tracker_kitti_output_dir does not exist: {self.tracker_kitti_output_dir}")
            return False
        return True

class ImageTensorInput(BufferProvider):

    def __init__(self, height, width, format):
        super().__init__()
        self.width = width
        self.height = height
        self.format = format
        self.framerate = 1
        self.device = 'cpu'
        self.queue = Queue(maxsize=10)

    def generate(self, size):
        tensor = self.queue.get()
        if isinstance(tensor, np.ndarray):
            return Buffer(tensor.tolist())
        elif isinstance(tensor, Stop):
            # EOS
            return Buffer()
        else:
            logger.exception("Unexpected input tensor data")
            return Buffer()

    def send(self, data):
        self.queue.put(data)

class GenericTensorInput():
    def __init__(self, device_id):
        self.queue = Queue(maxsize=10)
        self._device_id = device_id

    def generate(self, n):
        try:
            tensors = self.queue.get(timeout=1)
        except Empty:
            logger.warning("No tensor data to generate")
            return dict()
        result = {k: as_tensor(v, "").to_gpu(self._device_id) for k, v in tensors.items()}
        return result

    def send(self, data):
        self.queue.put(data)

class StaticTensorInput():
    def __init__(self, device_id):
        self._tensors = {}
        self._device_id = device_id

    def generate(self, n):
        result = {k: as_tensor(v, "").to_gpu(self._device_id) for k, v in self._tensors.items()}
        return result

    def set(self, data):
        self._tensors.update(data)

class TensorInputPool(ABC):
    @abstractmethod
    def submit(self, data: List):
        pass
    @abstractmethod
    def stop(self, reason: str):
        pass

class ImageTensorInputPool(TensorInputPool):

    def __init__(self, height, width, formats, batch_size, image_tensor_name, media_url_tensor_name, mime_tensor_name, device_id, require_extra_input):
        self._image_inputs = [ImageTensorInput(width, height, format) for format in formats for _ in range(batch_size)]
        self._media_url_tensor_name = media_url_tensor_name
        self._mime_tensor_name = mime_tensor_name
        self._image_tensor_name = image_tensor_name
        self._generic_input = GenericTensorInput(device_id) if require_extra_input else None
        self._batch_size = batch_size

    @property
    def image_inputs(self):
        return self._image_inputs

    @property
    def generic_input(self):
        return self._generic_input

    def submit(self, data: List):
        indices = []
        for item in data:
            mime_type = item.pop(self._mime_tensor_name, None)
            if mime_type is None:
                logger.error("MIME type is not specified")
                continue
            mime_type = mime_type.split('/');
            if mime_type[0] == 'image':
                format = mime_type[1].upper()
                # try find the free slot for the specific format
                i, image_tensor_input = next(((i, x) for i,x in enumerate(self._image_inputs) if x.format == format and x.queue.empty()), (-1, None))
                if image_tensor_input is None:
                    i, image_tensor_input = next(((i, x) for i,x in enumerate(self._image_inputs) if x.format == format), (-1, None))
                if image_tensor_input is not None:
                    if self._image_tensor_name in item:
                        image_tensor = item.pop(self._image_tensor_name)
                        image_tensor_input.send(image_tensor)
                        indices.append(i)
                    elif self._media_url_tensor_name in item:
                        image_url = item.pop(self._media_url_tensor_name)
                        with open(image_url, 'rb') as f:
                            image_tensor = np.frombuffer(f.read(), dtype=np.uint8)
                        image_tensor_input.send(image_tensor)
                        indices.append(i)
                    else:
                        logger.error(f"image tensor or media url is missing: {item}")
                else:
                    logger.error(f"Unable to find free slot for format {format}")
            else:
                logger.error(f"Unsupported MIME type {mime_type}")
                continue
        if self._generic_input:
            data = [data[i:i + self._batch_size] for i in range(0, len(data), self._batch_size)]
            for d in data:
                self._generic_input.send(stack_tensors_in_dict(d))
        # batched indices for each input
        return indices

    def stop(self, reason: str):
        for input in self._image_inputs:
            input.send(Stop(reason))
        if self._generic_input:
            self._generic_input.send(Stop(reason))

class BulkVideoInputPool(TensorInputPool):
    def __init__(self,
        max_batch_size,
        batch_timeout,
        media_url_tensor_name,
        source_tensor_name,
        mime_tensor_name,
        infer_config_paths,
        preprocess_config_paths,
        tracker_config: TrackerConfig,
        analytics_config: AnalyticsConfig,
        msgbroker_config: MessageBrokerConfig,
        render_config: RenderConfig,
        perf_config: PerfConfig,
        kitti_config: KittiConfig,
        output,
        device_id,
        require_extra_input,
        engine_file_names,
        dims,
        label_file_path,
        model_home
    ):
        self._batch_size = max_batch_size
        self._batch_timeout = batch_timeout
        self._media_url_tensor_name = media_url_tensor_name
        self._source_tensor_name = source_tensor_name
        self._mime_tensor_name = mime_tensor_name
        self._infer_config_paths = infer_config_paths
        self._engine_file_names = engine_file_names
        self._preprocess_config_paths = preprocess_config_paths
        self._pipeline = None
        self._output = output
        self._generic_input = StaticTensorInput(device_id) if require_extra_input else None
        self._device_id = device_id
        self._dims = dims
        self._tracker_config = tracker_config
        self._analytics_config = analytics_config
        self._msgbroker_config = msgbroker_config
        self._render_config = render_config
        self._perf_config = perf_config
        self._kitti_config = kitti_config
        self._label_file_path = label_file_path
        self._model_home = model_home

    def submit(self, data: List):
        def on_message(message, total_streams, output):
            # handle the dynamic source messages only in case the source config is set
            if not isinstance(message, DynamicSourceMessage):
                return
            logger.info(f"Received source added/removed message: {message.source_added}")
            if message.source_added:
                total_streams += 1
            else:
                total_streams -= 1
                if total_streams <= 0:
                    output.reset()

        url_list = []
        source_config_file = None
        total_streams = 0
        for item in data:
            if self._mime_tensor_name and self._mime_tensor_name in item:
                item.pop(self._mime_tensor_name)
            if self._media_url_tensor_name and self._media_url_tensor_name in item:
                url_list.append(str(item.pop(self._media_url_tensor_name)))
            elif self._source_tensor_name and self._source_tensor_name in item:
                source_config_file = item.pop(self._source_tensor_name)
                if not isinstance(source_config_file, str):
                    logger.error(
                        f"Source config value must be a string, got {type(source_config_file).__name__}: {source_config_file}"
                    )
                    return []
                if not source_config_file.lower().endswith(('.yml', '.yaml')):
                    logger.error(
                        f"Source config file must be a YAML file: {source_config_file}"
                    )
                    return []
                # Correct relative path to absolute path relative to model_home
                if not os.path.isabs(source_config_file):
                    source_config_file = os.path.join(self._model_home, source_config_file)
                break
            else:
                logger.error(f"Invalid input data: {data}")
                continue

        if len(url_list) > self._batch_size:
            logger.warning(
                f"Number of media urls ({len(url_list)}) > "
                f"batch size ({self._batch_size}), "
                f"only the first {self._batch_size} will be used"
            )
            url_list = url_list[:self._batch_size]

        pipeline = Pipeline(f"deepstream-video-batch")
        if self._generic_input and data:
            self._generic_input.set(stack_tensors_in_dict(data))

        # Use appropriate batch_capture method based on input type
        if source_config_file:
            sc = SourceConfig()
            sc.load(source_config_file)
            total_streams = len(sc.sensor_list)
            flow = Flow(pipeline).batch_capture(
                input=source_config_file,
                width=self._dims[1], height=self._dims[0],
                batch_size=self._batch_size,
                batched_push_timeout=self._batch_timeout
            )
        else:
            total_streams = len(url_list)
            flow = Flow(pipeline).batch_capture(
                url_list,
                width=self._dims[1],
                height=self._dims[0],
                batched_push_timeout=self._batch_timeout)

        for config in self._preprocess_config_paths:
            flow = flow.preprocess(config, None if not self._generic_input else self._generic_input.generate)
        for config_path, engine_file in zip(self._infer_config_paths, self._engine_file_names):
            if engine_file:
                flow = flow.infer(config_path, batch_size=self._batch_size, model_engine_file=engine_file)
            else:
                flow = flow.infer(config_path, batch_size=self._batch_size)
            if self._kitti_config.infer_kitti_output_dir:
                flow = flow.attach(what="kitti_dump_probe", name="inference_kitti_dump", properties={"kitti-dir": self._kitti_config.infer_kitti_output_dir})
        if self._tracker_config:
            flow = flow.track(
                ll_config_file=self._tracker_config.config_path,
                ll_lib_file=self._tracker_config.lib_path,
                gpu_id=self._device_id,
                tracker_width=self._tracker_config.width,
                tracker_height=self._tracker_config.height,
                display_tracking_id=self._tracker_config.display_tracking_id
            )
            if self._kitti_config.tracker_kitti_output_dir:
                flow = flow.attach(what="kitti_dump_probe", name="tracker_kitti_dump", properties={"tracker-kitti-output": True,"kitti-dir": self._kitti_config.tracker_kitti_output_dir})
        if self._analytics_config:
            flow = flow.analyze(
                self._analytics_config.config_path,
                enable=1,
                gpu_id=self._analytics_config.gpu_id
            )

        if self._perf_config.enable_fps_logs:
            flow = flow.attach(what="measure_fps_probe", name="fps_probe")
        if self._perf_config.enable_latency_logs:
            flow = flow.attach(what="measure_latency_probe", name="latency_probe")
        n_sink = 0
        if self._msgbroker_config:
            flow = flow.attach(
                what="add_message_meta_probe",
                name="message_generator",
                properties={
                    "label-file": self._label_file_path if self._label_file_path else "",
                    "source-config": source_config_file if source_config_file else ""
                }
            )
            flow = flow.fork()
            n_sink += 1
            publish_params = {
                'msg_broker_proto_lib': self._msgbroker_config.proto_lib_path,
                'msg_broker_conn_str': self._msgbroker_config.conn_str,
                'topic': self._msgbroker_config.topic,
                'msg_conv_config': self._msgbroker_config.msgconv_config_path,
                'msg_conv_payload_type': self._msgbroker_config.msgconv_payload_type,
                'msg_conv_msg2p_new_api': self._msgbroker_config.msgconv_msg2p_new_api,
                'msg_conv_frame_interval': self._msgbroker_config.msgconv_frame_interval,
                'sync': False
            }
            if self._msgbroker_config.msgconv_msg2p_lib:
                publish_params['msg_conv_msg2p_lib'] = self._msgbroker_config.msgconv_msg2p_lib

            flow.publish(**publish_params)
        if self._render_config.enable_stream:
            if n_sink == 0:
                flow = flow.fork()
                n_sink += 1
            flow.render(RenderMode.STREAM,
                       enable_osd=self._render_config.enable_osd,
                       rtsp_mount_point=self._render_config.rtsp_mount_point,
                       rtsp_port=self._render_config.rtsp_port,
                       sync=False)
        if self._render_config.enable_display:
            if n_sink == 0:
                flow = flow.fork()
                n_sink += 1
            seg_mask_config = dict(self._render_config.seg_mask_config) if self._render_config.seg_mask_config else None
            flow.render(RenderMode.DISPLAY, enable_osd=self._render_config.enable_osd, seg_mask_config=seg_mask_config, sync=False)

        flow = flow.attach(what=Probe("metadata_probe", self._output)).retrieve(self._output, **{'async': False})

        if self._pipeline is not None:
            self._pipeline.wait()

        if source_config_file:
            # monitor the total streams and reset the output when the total streams is 0
            pipeline.start(on_message=lambda message: on_message(message, total_streams, self._output))
        else:
            pipeline.start()
        self._pipeline = pipeline
        return list(range(len(url_list))) if url_list else list(range(self._batch_size))

    def stop(self, reason: str):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.wait()


class BaseTensorOutput(BufferRetriever, BatchMetadataOperator):
    """Base class for tensor outputs that can work as both BufferRetriever (appsink) and
    BatchMetadataOperator (probe).

    Subclasses override _extract_metadata() to yield results.
    Both consume() (appsink) and handle_metadata() (probe) iterate and deposit.
    """
    def __init__(self, n_outputs, metadata_output_name: str = None, image_output_name: str = None):
        """Initialize the base tensor output handler.

        Args:
            n_outputs: Maximum number of output slots (typically max_batch_size
                or max_batch_size * num_formats). Each slot corresponds to a
                pad index in the DeepStream pipeline.
            metadata_output_name: When set, collected results are wrapped in a
                dict keyed by this name (e.g. {"output": <result>}). Used by
                MetadataOutput / PreprocessMetadataOutput for DS metadata
                outputs (TYPE_CUSTOM_DS_METADATA). None for raw tensor outputs.
            image_output_name: When set, indicates that decoded image frames
                should be extracted from the DeepStream pipeline and returned
                as RGB uint8 tensors in HWC layout (TYPE_CUSTOM_DS_IMAGE
                output). The value is the output tensor name to use.
        """
        BufferRetriever.__init__(self)
        BatchMetadataOperator.__init__(self)
        self._n_outputs = n_outputs
        self._queue = Queue()
        self._stashed = []
        self._metadata_output_name = metadata_output_name
        self._image_output_name = image_output_name
        self._image_queue = Queue()  # separate queue for decoded image frames
        self._image_stashed = []  # stashed image batches for cross-collect reuse

    def consume(self, buffer):
        """Called when used as appsink retriever (BufferRetriever interface).

        When image output is enabled, extracts decoded RGB HWC frames from
        the buffer and converts them to CPU numpy arrays via dlpack before
        queuing. The dlpack conversion must happen here while the buffer's
        GPU memory is still valid — once consume() returns, the buffer is
        released and the GPU tensor data becomes stale.

        Obtains batch_meta from the buffer to get the batch_id → pad_index
        mapping, then calls buffer.extract(batch_id) for each frame.
        Note: batch_id is assigned by arrival order, not pad index — e.g. if
        only pads 0, 2, 3 deliver frames, they get batch_ids 0, 1, 2.
        """
        if self._image_output_name:
            received = [None] * self._n_outputs
            batch_meta = buffer.batch_meta
            for frame_meta in batch_meta.frame_items:
                frame = buffer.extract(frame_meta.batch_id)
                if frame is not None:
                    # Clone via dlpack while buffer GPU memory is still valid;
                    # clone() copies to a new GPU allocation so the tensor
                    # survives after the buffer is released.
                    frame = torch.utils.dlpack.from_dlpack(frame).clone()
                    received[frame_meta.pad_index] = (frame, frame_meta.buffer_pts)
            self._image_queue.put(received)
        return 1

    def handle_metadata(self, batch_meta):
        """Called when used as probe (BatchMetadataOperator interface)."""
        for metadata in self._extract_metadata(batch_meta):
            self._deposit(metadata)

    def _extract_metadata(self, batch_meta):
        """Override in subclasses to extract metadata.

        Yields a list (indexed by pad_index) where each non-None element is
        a dict containing the extracted data. The dict should include a
        "timestamp" key (buffer_pts) for synchronization with image frames
        in collect(). DS metadata dicts already carry this; tensor outputs
        insert it explicitly.
        """
        return
        yield  # Make this a generator

    def collect(self, indices: List, timeout=None) -> List | None:
        def move_data(data: list, collected: list):
            for i, d in enumerate(data):
                if d is not None:
                    if collected[i] is None:
                        collected[i] = d
                        data[i] = None

        def _drain(queue, stashed, collected, indices):
            """Drain a queue into collected slots using stash for leftovers."""
            while stashed:
                data = stashed.pop(0)
                move_data(data, collected)
            while not all(collected[i] is not None for i in indices):
                try:
                    data = queue.get(timeout=timeout)
                    if data is None:
                        break
                    move_data(data, collected)
                    if any(d is not None for d in data):
                        stashed.append(data)
                except Empty:
                    break

        def _drain_images(queue, stashed, collected, indices, target_pts):
            """Drain image queue, discarding images older than metadata timestamps."""
            def process(data):
                for i, d in enumerate(data):
                    if d is None:
                        continue
                    # Always discard stale images even if this pad is already collected
                    if i in target_pts:
                        _, img_pts = d
                        if img_pts < target_pts[i]:
                            data[i] = None
                            continue
                    if collected[i] is None:
                        collected[i] = d
                        data[i] = None

            remaining = []
            for data in stashed:
                process(data)
                if any(d is not None for d in data):
                    remaining.append(data)
            stashed.clear()
            stashed.extend(remaining)

            while not all(collected[i] is not None for i in indices):
                try:
                    data = queue.get(timeout=timeout)
                    if data is None:
                        break
                    process(data)
                    if any(d is not None for d in data):
                        stashed.append(data)
                except Empty:
                    break

        n_slots = max(indices) + 1

        # Drain metadata queue first — metadata drives collection.
        meta_collected = [None] * n_slots
        _drain(self._queue, self._stashed, meta_collected, indices)

        if self._image_output_name:
            # Build target timestamps from metadata for image synchronization.
            # Images with pts older than the metadata are stale and discarded.
            target_pts = {}
            for i in indices:
                meta = meta_collected[i]
                if meta is not None and isinstance(meta, dict):
                    pts = meta.get("timestamp")
                    if pts is not None:
                        target_pts[i] = pts

            image_collected = [None] * n_slots
            _drain_images(self._image_queue, self._image_stashed, image_collected, indices, target_pts)

            if all(i is None for i in meta_collected) and all(i is None for i in image_collected):
                return None

            results = []
            for idx in indices:
                result = {}
                metadata = meta_collected[idx]
                image_entry = image_collected[idx]

                if metadata is not None:
                    if self._metadata_output_name is not None:
                        result[self._metadata_output_name] = metadata
                    elif isinstance(metadata, dict):
                        result.update(metadata)
                    else:
                        result = metadata

                if image_entry is not None:
                    frame, _ = image_entry
                    if not isinstance(result, dict):
                        result = {}
                    result[self._image_output_name] = frame

                results.append(result if result else None)
            return results
        else:
            # --- Metadata-only path ---
            if all(i is None for i in meta_collected):
                return None
            collected = [meta_collected[i] for i in indices]
            return collected if self._metadata_output_name is None else [
                {self._metadata_output_name: r} for r in collected
            ]

    def reset(self):
        # wake up the old queue first
        self._queue.put(None)
        self._queue = Queue(maxsize=self._queue.maxsize)
        self._stashed = []
        self._image_queue = Queue()
        self._image_stashed = []

    def _deposit(self, received: list):
        logger.debug("DeepstreamBackend: Depositing data: %s", received)
        try:
            self._queue.put(received)
        except Full:
            logger.warning(f"DeepstreamBackend: Queue is full, dropping data: {received}")


class TensorOutput(BaseTensorOutput):
    def __init__(self, n_outputs, preprocess_config_path, image_output_name=None):
        super().__init__(n_outputs, metadata_output_name=None, image_output_name=image_output_name)
        self._preprocess_config_path = preprocess_config_path

    def _extract_metadata(self, batch_meta):
        received = [None] * self._n_outputs
        if self._preprocess_config_path:
            for meta in batch_meta.preprocess_batch_items:
                preprocess_batch = meta.as_preprocess_batch()
                if not preprocess_batch:
                    continue
                for roi in preprocess_batch.rois:
                    result = dict()
                    for user_meta in roi.tensor_items:
                        tensor_output = user_meta.as_tensor_output()
                        if tensor_output :
                            for n, tensor in tensor_output.get_layers().items():
                                torch_tensor = torch.utils.dlpack.from_dlpack(tensor).to('cpu')
                                result[n] = torch_tensor
                    result["timestamp"] = roi.frame_meta.buffer_pts
                    received[roi.frame_meta.pad_index] = result
        else:
            for frame_meta in batch_meta.frame_items:
                result = dict()
                for user_meta in frame_meta.tensor_items:
                    tensor_output = user_meta.as_tensor_output()
                    if tensor_output :
                        for n, tensor in tensor_output.get_layers().items():
                            torch_tensor = torch.utils.dlpack.from_dlpack(tensor).to('cpu')
                            result[n] = torch_tensor
                result["timestamp"] = frame_meta.buffer_pts
                received[frame_meta.pad_index] = result
        yield received

class PreprocessMetadataOutput(BaseTensorOutput):
    """Output handler for preprocess metadata."""
    def __init__(self, n_outputs, output_name, dims, image_output_name=None):
        super().__init__(n_outputs, metadata_output_name=output_name, image_output_name=image_output_name)
        self._n_outputs = n_outputs
        self._shape = dims

    def _extract_metadata(self, batch_meta):
        for meta in batch_meta.preprocess_batch_items:
            preprocess_batch = meta.as_preprocess_batch()
            if not preprocess_batch:
                continue
            received = preprocess_batch.extract(self._n_outputs)
            for roi in preprocess_batch.rois:
                idx = roi.frame_meta.pad_index
                if received[idx] is not None:
                    received[idx]["timestamp"] = roi.frame_meta.buffer_pts
            yield received

class MetadataOutput(BaseTensorOutput):
    def __init__(self, n_outputs, output_name, dims, image_output_name=None):
        super().__init__(n_outputs, metadata_output_name=output_name, image_output_name=image_output_name)
        self._n_outputs = n_outputs
        self._shape = dims

    def _extract_metadata(self, batch_meta):
        # DS metadata from batch_meta.extract() already carries pts internally
        yield batch_meta.extract(self._n_outputs)

class DeepstreamBackend(ModelBackend):
    """Deepstream backend using pyservicemaker"""
    def __init__(self, model_config:Dict, model_home: str, device_id: int=0):
        super().__init__(model_config, model_home, device_id)
        self._max_batch_size = model_config["max_batch_size"]
        self._model_name = model_config["name"]
        self._output_names = [o['name'] for o in model_config['output']]
        self._output_types = [o['data_type'] for o in model_config['output']]
        self._image_tensor_name = None
        self._media_url_tensor_name = None
        self._mime_tensor_name = None
        self._source_tensor_name = None
        self._image_output_name = None
        self._inference_timeout = model_config["parameters"].get("inference_timeout", None)
        self._in_pools = {}
        self._outputs = {}
        self._pipelines = {}

        tensor_output = False if self._output_types[0] == "TYPE_CUSTOM_DS_METADATA" else True
        for o in model_config['output']:
            if o['data_type'] == 'TYPE_CUSTOM_DS_IMAGE':
                self._image_output_name = o['name']
                break
        dims = (0, 0)

        if "parameters" not in model_config or "infer_config_path" not in model_config["parameters"]:
            raise Exception("Deepstream pipeline requires infer_config_path")

        infer_config_paths = self._correct_config_paths(model_config["parameters"]['infer_config_path'])
        if not infer_config_paths:
            raise Exception("Deepstream pipeline requires infer_config_path")

        preprocess_config_paths = []
        if "preprocess_config_path" in model_config["parameters"]:
            preprocess_config_paths = self._correct_config_paths(model_config["parameters"]['preprocess_config_path'])

        if "tracker_config" in model_config["parameters"]:
            tracker_config = TrackerConfig(
                config_path=self._correct_config_paths(
                    [model_config["parameters"]["tracker_config"].get("ll_config_file")]
                )[0] if model_config["parameters"]["tracker_config"].get("ll_config_file") else None,
                lib_path=self._correct_config_paths(
                    [model_config["parameters"]["tracker_config"]["ll_lib_file"]]
                )[0],
                width=model_config["parameters"]["tracker_config"].get("width", 1920),
                height=model_config["parameters"]["tracker_config"].get("height", 1088),
                display_tracking_id=model_config["parameters"]["tracker_config"].get("display_tracking_id", False)
            )
            if not tracker_config:
                logger.warning("DeepstreamBackend: tracker_config is not properlyconfigured")
        else:
            tracker_config = TrackerConfig()

        if "msgbroker_config" in model_config["parameters"]:
            msgbroker_config = MessageBrokerConfig(
                proto_lib_path=self._correct_config_paths(
                    [model_config["parameters"]["msgbroker_config"]["msgbroker_proto_lib_path"]]
                )[0],
                conn_str=model_config["parameters"]["msgbroker_config"]["msgbroker_conn_str"],
                topic=model_config["parameters"]["msgbroker_config"]["msgbroker_topic"],
                msgconv_config_path=self._correct_config_paths(
                    [model_config["parameters"]["msgbroker_config"]["msgconv_config_path"]]
                )[0]
            )
            if "msgconv_payload_type" in model_config["parameters"]["msgbroker_config"]:
                msgbroker_config.msgconv_payload_type = model_config["parameters"]["msgbroker_config"]["msgconv_payload_type"]
            if "msgconv_msg2p_new_api" in model_config["parameters"]["msgbroker_config"]:
                msgbroker_config.msgconv_msg2p_new_api = model_config["parameters"]["msgbroker_config"]["msgconv_msg2p_new_api"]
            if "msgconv_frame_interval" in model_config["parameters"]["msgbroker_config"]:
                msgbroker_config.msgconv_frame_interval = model_config["parameters"]["msgbroker_config"]["msgconv_frame_interval"]
            if "msgconv_msg2p_lib" in model_config["parameters"]["msgbroker_config"]:
                msgbroker_config.msgconv_msg2p_lib = self._correct_config_paths(
                    [model_config["parameters"]["msgbroker_config"]["msgconv_msg2p_lib"]]
                )[0]
            if not msgbroker_config:
                logger.warning("DeepstreamBackend: msgbroker_config is not properly configured")
        else:
            msgbroker_config = MessageBrokerConfig()

        if "render_config" in model_config["parameters"]:
            render_config = RenderConfig(
                enable_display=model_config["parameters"]["render_config"].get("enable_display", False),
                enable_osd=model_config["parameters"]["render_config"].get("enable_osd", False),
                enable_stream=model_config["parameters"]["render_config"].get("enable_stream", False),
                rtsp_mount_point=model_config["parameters"]["render_config"].get("rtsp_mount_point", "/ds-test"),
                rtsp_port=model_config["parameters"]["render_config"].get("rtsp_port", 8554),
                seg_mask_config=model_config["parameters"]["render_config"].get("seg_mask_visualization", None)
            )
        else:
            render_config = RenderConfig()

        if "perf_config" in model_config["parameters"]:
            perf_config = PerfConfig(
                enable_fps_logs=model_config["parameters"]["perf_config"].get("enable_fps_logs", False),
                enable_latency_logs=model_config["parameters"]["perf_config"].get("enable_latency_logs", False)
            )
        else:
            perf_config = PerfConfig()

        if "kitti_output_path" in model_config["parameters"]:
            kitti_config = KittiConfig(
                infer_kitti_output_dir=model_config["parameters"]["kitti_output_path"].get("infer", None),
                tracker_kitti_output_dir=model_config["parameters"]["kitti_output_path"].get("tracker", None)
            )
            if not kitti_config:
                logger.warning("DeepstreamBackend: kitti_config is not properly configured")
        else:
            kitti_config = KittiConfig()

        if "analytics_config" in model_config["parameters"]:
            ac = model_config["parameters"]["analytics_config"]
            config_path = self._correct_config_paths([ac["config_file_path"]])[0] if ac.get("config_file_path") else None
            analytics_config = AnalyticsConfig(
                config_path=config_path,
                gpu_id=ac.get("gpu_id", device_id)
            )
            if not analytics_config:
                logger.warning("DeepstreamBackend: analytics_config is not properly configured")
        else:
            analytics_config = AnalyticsConfig()

        infer_element = model_config['backend'].split('/')[-1]
        with_triton = infer_element == 'nvinferserver'
        require_extra_input = False
        warmup_data_0 = dict()
        warmup_data_1 = dict()
        for input in model_config['input']:
            if input['data_type'] == 'TYPE_CUSTOM_DS_IMAGE':
                self._image_tensor_name = input['name']
            elif input['data_type'] == 'TYPE_CUSTOM_DS_MEDIA_URL':
                self._media_url_tensor_name = input['name']
            elif input['data_type'] == 'TYPE_CUSTOM_DS_MIME':
                self._mime_tensor_name = input['name']
            elif input['data_type'] == 'TYPE_CUSTOM_DS_SOURCE_CONFIG':
                self._source_tensor_name = input['name']
            elif not ('optional' in input and input['optional']):
                tensor_name = input['name']
                require_extra_input = True
                np_type = np_datatype_mapping[input['data_type']]
                warmup_data_0[tensor_name] = np.random.rand(*input['dims']).astype(np_type)
                warmup_data_1[tensor_name] = np.random.rand(*input['dims']).astype(np_type)
        if (self._image_tensor_name is None and
            self._media_url_tensor_name is None and
            self._source_tensor_name is None):
            raise ValueError(
                "Deepstream pipeline requires at least one "
                "TYPE_CUSTOM_DS_IMAGE or TYPE_CUSTOM_DS_MEDIA_URL input "
                "or TYPE_CUSTOM_DS_SOURCE_CONFIG input"
            )
        if ((self._image_tensor_name or self._media_url_tensor_name) and
            self._mime_tensor_name is None):
            raise ValueError("Deepstream pipeline requires TYPE_CUSTOM_DS_MIME input")

        primary_infer_config_property = {}
        try:
            primary_infer_config_path = infer_config_paths[0]
            with open(primary_infer_config_path, 'r') as f:
                primary_infer_config = yaml.safe_load(f)
            if "property" in primary_infer_config:
                primary_infer_config_property = primary_infer_config["property"]
        except Exception as e:
            raise RuntimeError(f"Failed to load primary inference config: {e}") from e

        if "resize_video" in model_config["parameters"] and len(model_config["parameters"]["resize_video"]) == 2:
            resize_to = model_config["parameters"]["resize_video"]
            dims = (resize_to[0], resize_to[1])
            logger.info(f"DeepstreamBackend: setting video size to {dims}")
        else:
            # video resized to network dimensions from  the primary inference config by default
            if "infer-dims" in primary_infer_config_property:
                infer_dims = [int(dim) for dim in primary_infer_config_property["infer-dims"].split(";")]
                if len(infer_dims) == 3:
                    if "network-input-order" in primary_infer_config_property and primary_infer_config_property["network-input-order"] == 1:
                        dims = (infer_dims[0], infer_dims[1])
                    else:
                        dims = (infer_dims[1], infer_dims[2])
                    logger.info(f"DeepstreamBackend: setting video size to network dimensions: {dims}")

        if dims[0] == 0 or dims[1] == 0:
            raise ValueError(
                "DeepstreamBackend: unable to find network dimensions: "
                "infer-dims missing in the config?"
            )

        # get the label file path from the primary inference config
        label_file_path = None
        if "labelfile-path" in primary_infer_config_property:
            label_path = primary_infer_config_property["labelfile-path"]
            # Convert to absolute path if it's relative
            if not os.path.isabs(label_path):
                label_file_path = os.path.join(self._model_home, label_path)
            else:
                label_file_path = label_path

        # construct the input pools, outputs and pipelines
        if self._image_tensor_name is not None:
            # image input support
            media = "image"
            formats = ["JPEG", "PNG"]
            in_pool = ImageTensorInputPool(dims[0], dims[1], formats, self._max_batch_size, self._image_tensor_name, self._media_url_tensor_name, self._mime_tensor_name, device_id, require_extra_input)
            n_output = self._max_batch_size * len(formats)
            if tensor_output:
                output = TensorOutput(n_output, preprocess_config_paths, image_output_name=self._image_output_name)
            elif preprocess_config_paths:
                output = PreprocessMetadataOutput(n_output, self._output_names[0], dims, image_output_name=self._image_output_name)
            else:
                output = MetadataOutput(n_output, self._output_names[0], dims, image_output_name=self._image_output_name)
            # create the pipeline
            pipeline = Pipeline(f"deepstream-{self._model_name}-{media}")

            self._pipelines[media] = pipeline
            self._in_pools[media] = in_pool
            self._outputs[media] = output

            # build the inference flow
            flow = Flow(pipeline)
            batch_timeout = model_config["parameters"].get("batch_timeout", 1000 * self._max_batch_size)
            flow = flow.inject(in_pool.image_inputs).decode().batch(batch_size=self._max_batch_size, batched_push_timeout=batch_timeout, live_source=False, width=dims[1], height=dims[0])
            for config_file in preprocess_config_paths:
                input = self._in_pools[media].generic_input if require_extra_input else None
                flow = flow.preprocess(config_file, None if not input else input.generate)
            for config_file in infer_config_paths:
                engine_file = self._generate_engine_name(config_file, device_id, self._max_batch_size)
                if engine_file:
                    flow = flow.infer(config_file, with_triton, batch_size=self._max_batch_size, model_engine_file=engine_file)
                else:
                    flow = flow.infer(config_file, with_triton, batch_size=self._max_batch_size)

            flow = flow.attach(what=Probe("metadata_probe", output)).retrieve(output, **{'async': False})

            # warm up
            warmup_data_0[self._image_tensor_name] = np.frombuffer(png_data, dtype=np.uint8)
            warmup_data_0[self._mime_tensor_name] = "image/png"
            warmup_data_1[self._image_tensor_name] = np.frombuffer(jpg_data, dtype=np.uint8)
            warmup_data_1[self._mime_tensor_name] = "image/jpeg"

            logger.info("Image decoder warming up - pre-filling all queues")
            # Submit BOTH PNG and JPEG warmup data BEFORE starting pipeline
            # This ensures all 4 sources have data when streaming threads start
            indices_0 = in_pool.submit([warmup_data_0.copy() for _ in range(self._max_batch_size)])
            indices_1 = in_pool.submit([warmup_data_1.copy() for _ in range(self._max_batch_size)])

            # Now start pipeline - all queues have data
            pipeline.start()

            # Collect all results together (PNG + JPEG)
            all_indices = indices_0 + indices_1
            results = output.collect(all_indices)
            output.reset()
            logger.info(f"Warm up done: {results}")

        if (self._media_url_tensor_name is not None or
            self._source_tensor_name is not None):
            # video input support
            media = "video"
            if tensor_output:
                output = TensorOutput(self._max_batch_size, preprocess_config_paths, image_output_name=self._image_output_name)
            elif preprocess_config_paths:
                output = PreprocessMetadataOutput(self._max_batch_size, self._output_names[0], dims, image_output_name=self._image_output_name)
            else:
                output = MetadataOutput(self._max_batch_size, self._output_names[0], dims, image_output_name=self._image_output_name)
            engine_files = [
                self._generate_engine_name(
                    config_file,
                    device_id,
                    self._max_batch_size
                )
                for config_file in infer_config_paths
            ]
            in_pool = BulkVideoInputPool(
                max_batch_size=self._max_batch_size,
                batch_timeout=model_config["parameters"].get("batch_timeout", 1000 * self._max_batch_size),
                media_url_tensor_name=self._media_url_tensor_name,
                source_tensor_name=self._source_tensor_name,
                mime_tensor_name=self._mime_tensor_name,
                infer_config_paths=infer_config_paths,
                preprocess_config_paths=preprocess_config_paths,
                tracker_config=tracker_config,
                analytics_config=analytics_config,
                msgbroker_config=msgbroker_config,
                render_config=render_config,
                perf_config=perf_config,
                kitti_config=kitti_config,
                output=output,
                device_id=device_id,
                require_extra_input=require_extra_input,
                engine_file_names=engine_files,
                dims=dims,
                label_file_path=label_file_path,
                model_home=self._model_home
            )

            # build tensorrt engine if not exist
            if not all (os.path.exists(e) for e in engine_files):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    f.write(jpg_data)
                    warmup_video_path = f.name
                # Disable perf/render probes for warmup to avoid leaked FPS reporters
                in_pool._perf_config = PerfConfig()
                in_pool._render_config = RenderConfig()
                try:
                    logger.info("Video decoder warming up - building TensorRT engine")
                    warmup_item = {self._media_url_tensor_name: warmup_video_path}
                    if self._mime_tensor_name:
                        warmup_item[self._mime_tensor_name] = "video/mp4"
                    indices = in_pool.submit([warmup_item])
                    if indices:
                        results = output.collect(indices)
                        logger.info(f"Video warm up done: {results}")
                    output.reset()
                    in_pool.stop("warmup")
                finally:
                    in_pool._perf_config = perf_config
                    in_pool._render_config = render_config
                    os.unlink(warmup_video_path)

        self._in_pools[media] = in_pool
        self._outputs[media] = output

        logger.info(
            f"DeepstreamBackend created for {self._model_name} to generate "
            f"{self._output_names}, output tensor: {tensor_output}"
        )


    def __call__(self, *args, **kwargs):
        in_data_list = args if args else split_tensor_in_dict(kwargs)
        media = None
        explicit_batch = True if args else False

        # analyze the input batch
        for data in in_data_list:
            # convert numpy types to native Python types
            for k, v in data.items():
                if isinstance(v, np.generic):
                    v = v.item()
                if isinstance(v, bytes):
                    v = v.decode()
                data[k] = v
            # get the media type
            if ((self._image_tensor_name in data or self._media_url_tensor_name in data) and
                not self._mime_tensor_name in data):
                logger.error(f"MIME type is not specified for input {data}")
                return
            if self._source_tensor_name and self._source_tensor_name in data:
                media = "video"
                explicit_batch = True
                break # only video source is supported through source config

            current_media = data[self._mime_tensor_name].split('/')[0]
            if media is None:
                media = current_media
            elif media != current_media:
                logger.error(
                    f"Mixed media types are not supported in a single batch, "
                    f"got {media} and {current_media}"
                )
                return

        # validate the media type
        if media not in self._in_pools:
            if media == "image":
                logger.error("TYPE_CUSTOM_DS_IMAGE input must be added to the pipeline for image support")
            elif media == "video":
                logger.error("TYPE_CUSTOM_DS_MEDIA_URL input must be added to the pipeline for video support")
            else:
                logger.error("Unknown media type : ", media)
            return

        # submit the data to the pipeline which supports the media type
        indices = self._in_pools[media].submit(in_data_list)
        # collect the results
        while True:
            # TODO: timeout should be runtime configurable
            results = self._outputs[media].collect(indices, timeout=self._inference_timeout)
            if not results:
                logger.info("DeepstreamBackend: No more data from this batch")
                break
            out_data_list = []
            for result in results:
                out_data = dict()
                for o in self._output_names:
                    if o in result:
                        out_data[o] = result[o]
                    else:
                        out_data[o] = None
                out_data_list.append(out_data)
            yield out_data_list if explicit_batch else out_data_list[0]
            # No consecutive inference results for image
            if media == "image":
                break

    def stop(self):
        for input_pool in self._in_pools.values():
            input_pool.stop("Finalized")
        for pipeline in self._pipelines.values():
            pipeline.stop()
            pipeline.wait()
        self._in_pools.clear()
        self._pipelines.clear()
        super().stop()

    def __del__(self):
        self.stop()

    def _generate_engine_name(self, config_path: str, device_id: int, batch_size: int):
        def network_mode_to_string(network_mode: int):
            if network_mode == 0:
                return "fp32"
            elif network_mode == 1:
                return "int8"
            elif network_mode == 2:
                return "fp16"
            else:
                return ""
        network_mode = "fp16"
        onnx_file = "model.onnx"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not "property" in config:
                return None
            property = config["property"]
            if not "onnx-file" in property:
                return None
            onnx_file = property["onnx-file"]
            if "network-mode" in property:
                mode = network_mode_to_string(property["network-mode"])
                if mode:
                    network_mode = mode
        engine_file = f"{onnx_file}_b{batch_size}_gpu{device_id}_{network_mode}.engine"
        return os.path.join(self._model_home, engine_file)


    def _correct_config_paths(self, config_paths: List[str]) -> List[str]:
        if not config_paths:
            return []
        for i, path in enumerate(config_paths):
            if not os.path.isabs(path):
                config_paths[i] = os.path.join(self._model_home, path)
        return config_paths
