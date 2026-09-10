# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import platform
from enum import Enum
from collections import namedtuple
from urllib.parse import urlparse
from ._pydeepstream import Receiver, BufferRetriever, Feeder, BatchMetadataOperator, Tensor, Probe, PreprocessBatchUserMetadata, CommonFactory, signal
from .pipeline import Pipeline
from .source_config import SourceConfig, RecordConfig
from typing import Callable, Optional

class MediaType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"


class RenderMode(Enum):
    DISPLAY = 1
    DISCARD = 2
    STREAM = 3


StreamInfo = namedtuple("StreamInfo", ["originator", "template", "media_type"])
StreamInfo.__new__.__defaults__ = ("", "generic")

_mime_table = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "RGB": "video/x-raw",
    "RGBA": "video/x-raw",
    "I420": "video/x-raw",
    "U8": "audio/x-raw",
    "S8": "audio/x-raw",
    "U16LE": "audio/x-raw",
    "U16BE": "audio/x-raw",
    "S16LE": "audio/x-raw",
    "S16BE": "audio/x-raw",
    "U24LE": "audio/x-raw",
    "U24BE": "audio/x-raw",
    "S24LE": "audio/x-raw",
    "S24BE": "audio/x-raw",
    "U32LE": "audio/x-raw",
    "U32BE": "audio/x-raw",
    "S32LE": "audio/x-raw",
    "S32BE": "audio/x-raw",
    "F32LE": "audio/x-raw",
    "F32BE": "audio/x-raw",
    "F64LE": "audio/x-raw",
    "F64BE": "audio/x-raw"
}
_raw_video_formats = {"RGB", "RGBA", "I420"}
_raw_audio_formats = {
    "U8", "S8",
    "U16LE", "U16BE", "S16LE", "S16BE",
    "U24LE", "U24BE", "S24LE", "S24BE",
    "U32LE", "U32BE", "S32LE", "S32BE",
    "F32LE", "F32BE", "F64LE", "F64BE",
}
_default_audio_msg2p_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_msgconv_audio.so"

def _populate_kwargs(properties, **kwargs):
    for key, value in kwargs.items():
        properties[key.replace('_', '-')] = value

def _infer_media_type(template: str):
    if template.startswith("asrc_") or template.startswith("audio_"):
        return "audio"
    if template.startswith("vsrc_") or template.startswith("video_"):
        return "video"
    return "generic"

def _normalize_media_types(media_types):
    if isinstance(media_types, str):
        media_types = [media_types]
    elif media_types is None:
        media_types = ["video"]
    else:
        media_types = list(media_types)
    if not media_types:
        raise TypeError("At least one media type must be supplied")

    normalized = []
    for media_type in media_types:
        if isinstance(media_type, MediaType):
            media_type = media_type.value
        elif not isinstance(media_type, str):
            raise TypeError(f"Unsupported media type: {media_type}")
        media_type = media_type.lower()
        if media_type in ["av", "both", "all"]:
            for expanded in ["video", "audio"]:
                if expanded not in normalized:
                    normalized.append(expanded)
            continue
        if media_type not in ["video", "audio"]:
            raise TypeError(f"Unsupported media type: {media_type}")
        if media_type not in normalized:
            normalized.append(media_type)
    return normalized

def _parse_stream_info(info):
    if isinstance(info, StreamInfo):
        return info

    items = info.split('/', 1)
    template = items[1] if len(items) == 2 else ''
    return StreamInfo(
        originator=items[0],
        template=template,
        media_type=_infer_media_type(template)
    )

def _link_stream(pipeline: Pipeline, stream_info: StreamInfo, target: str):
    if stream_info.template:
        pipeline.link((stream_info.originator, target), (stream_info.template, ""))
    else:
        pipeline.link(stream_info.originator, target)

def _nvmm_audio_caps(audio_caps: str):
    if not audio_caps:
        return None
    caps = audio_caps
    if "memory:NVMM" not in caps:
        caps = caps.replace("audio/x-raw", "audio/x-raw(memory:NVMM)", 1)
    defaults = {
        "format=": "format=S16LE",
        "layout=": "layout=interleaved",
        "channels=": "channels=1",
    }
    for marker, value in defaults.items():
        if marker not in caps:
            caps = f"{caps}, {value}"
    return caps

def _add_first_available_element(pipeline: Pipeline, type_names, name: str, properties=None) -> str:
    candidates = [type_names] if isinstance(type_names, str) else list(type_names)
    last_error = None
    for type_name in candidates:
        try:
            pipeline.add(type_name, name, properties)
            return type_name
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(
        f"Element creation failed for candidates {', '.join(candidates)}"
    ) from last_error

def _default_audio_parser_for_encoder(encoder_name: str):
    if encoder_name == "opusenc":
        return "opusparse"
    if encoder_name in {"voaacenc", "avenc_aac", "faac", "fdkaacenc"}:
        return "aacparse"
    return None

def _normalize_rtsp_mount_point(mount_point: str) -> str:
    mount_point = (mount_point or "").strip()
    if not mount_point:
        return "/"
    return mount_point if mount_point.startswith("/") else f"/{mount_point}"

def _get_node_name(func, name) -> str:
    if not hasattr(func, "name_counter"):
        func.name_counter = 0
    else:
        func.name_counter += 1
    base_name = func.__name__.split('.')[-1]
    return f"{base_name}-{name}-{func.name_counter}"

def _validate_uri(uri: str):
    # parse the source uri
    parsed_uri = urlparse(uri)
    if not parsed_uri.scheme or parsed_uri.scheme == 'file':
        if not os.path.exists(parsed_uri.path):
            raise Exception(f"File {parsed_uri.path} doesn't exist")
        if not parsed_uri.scheme:
           uri = f"file://{parsed_uri.path}"
    return uri


class Flow:
    """
    The Flow class for constructing AI streaming flow
    """
    DEFAULT_PICTURE_WIDTH = 1920
    DEFAULT_PICTURE_HEIGHT = 1080
    DEFAULT_SOURCE_TYPE = "nvurisrcbin"
    DYNAMIC_SOURCE_TYPE = "nvdsdynamicsrcbin"
    DEFAULT_BATCH_PUSH_TIMEOUT = 33000

    def __init__(self, pipeline: Pipeline, streams=None, parent=None):
        """
        Initialize a new flow

        Args:
            pipeline(Pipeline): pipeline instance used by the flow
            streams(list[str]): list of stream descriptions
            parent(Flow):       parent flow from which the current flow is generated
        """
        self._pipeline = pipeline
        self._streams = [_parse_stream_info(stream) for stream in (streams or [])]
        self._parent = parent
        self._batch_size = parent._batch_size if parent else 0
        self._sr_controller = None

    def __call__(self, on_message=None):
        """
        Activate the current flow.
        The call will not return until the flow is done
        Args:
            on_message (Callable[Pipeline, PipelineMessage]): message callback, optional
        """
        self._pipeline.start(on_message).wait()

    @property
    def pipeline(self):
        return self._pipeline

    def capture(self, inputs: list, source_manager=None, media_types=(MediaType.VIDEO,), **kwargs):
        """
        Create a flow to capture a list of inputs, which creates one or more streams for
        any derived flow.

        Args:
            input(list(str)): list of input URIs
            media_types(MediaType, str, or list): requested stream kinds: video, audio, or both
            kwargs:           key word argument for the source node
        Return: A derived flow
        Raises: TypeError
        """
        func = Flow.capture
        streams = []
        media_types = _normalize_media_types(media_types)
        if source_manager is None:
            if not inputs:
                raise TypeError("No input list provided for capturing")
            # create source bin for each input in the list
            for input in inputs:
                source = _get_node_name(func, 'source')
                source_properties = {"uri": _validate_uri(input)}
                _populate_kwargs(source_properties, **kwargs)
                if "audio" in media_types:
                    source_properties["disable-audio"] = False
                self._pipeline.add(self.DEFAULT_SOURCE_TYPE, source, source_properties)
                for media_type in media_types:
                    template = "asrc_%u" if media_type == "audio" else "vsrc_%u"
                    streams.append(StreamInfo(source, template, media_type))
        else:
            if "audio" in media_types:
                raise NotImplementedError(
                    "Dynamic audio capture requires nvdsdynamicsrcbin audio pad support; "
                    "the current dynamic source bin links only video pads"
                )
            # create dynamic source bin
            source = _get_node_name(func, 'source')
            source_properties = dict()
            _populate_kwargs(source_properties, **kwargs)
            self._pipeline.add(self.DYNAMIC_SOURCE_TYPE, source, source_properties)
            source_manager.attach("add-source", self._pipeline[source])
            source_manager.attach("remove-source", self._pipeline[source])
            source_manager.attach("terminate", self._pipeline[source])
            for input in inputs:
                source_manager.add_source(input)
            streams.append(StreamInfo(source, media_type="video"))
        return Flow(self._pipeline, streams=streams, parent=self)

    def batch_capture(self, input, record_config: Optional[RecordConfig] = None, media_types=(MediaType.VIDEO,), **kwargs):
        """
        Create a flow to capture multiple sources and batch them together

        Args:
            input(list(str) or str): list of URIs or a source config file
            record_config(RecordConfig): unified recording configuration (local or cloud)
            media_types(MediaType, str, or list): requested stream kind: video or audio
            **kwargs: width, height, batch-size are supported
        Return: A derived flow
        Raises: TypeError
        """
        func = Flow.batch_capture
        streams = []
        source_config = None
        uri_list = []
        properties = {}
        default_mux_properties = {
            "width": self.DEFAULT_PICTURE_WIDTH,
            "height": self.DEFAULT_PICTURE_HEIGHT,
            "batch-size": 1,
            "batched-push-timeout": self.DEFAULT_BATCH_PUSH_TIMEOUT,
            "buffer-pool-size": 4,
            "drop-pipeline-eos": False,
            "live-source": False,
        }
        default_source_properties = {
            "file-loop": False
        }

        media_types = _normalize_media_types(media_types)
        if len(media_types) != 1:
            raise TypeError("batch_capture() supports exactly one media type per batched stream")
        batch_media_type = media_types[0]
        is_audio_batch = batch_media_type == "audio"
        record_audio = record_config and record_config.rec_mode in (0, 2)

        _populate_kwargs(properties, **kwargs)
        gpu_id = properties["gpu-id"] if "gpu-id" in properties else 0
        source = _get_node_name(func, 'source')

        if isinstance(input, list):
            uri_list = input
        elif isinstance(input, str) and os.path.splitext(input)[1] in ['.yaml', '.yml']:
            source_config = SourceConfig()
            source_config.load(input)
            uri_list = [sensor.uri for sensor in source_config.sensor_list]
        else:
            raise TypeError("input must be a source list or source config file")

        if len(uri_list) > 0:
            # default batch size is the number of sources
            default_mux_properties["batch-size"] = len(uri_list)

        if source_config and source_config.source_type == "nvmultiurisrcbin":
            source_properties = dict(source_config.source_properties)
            default_properties = default_source_properties | default_mux_properties
            for k, v in default_properties.items():
                if k in properties:
                    # override the source config properties
                    source_properties[k] = properties[k]
                elif not k in source_properties:
                    # use the default value
                    source_properties[k] = v
            batch_size = source_properties.pop("batch-size")
            if not "max-batch-size" in source_properties:
                source_properties["max-batch-size"] = batch_size
            source_properties["uri-list"] = ','.join([source.uri for source in source_config.sensor_list])
            source_properties["sensor-id-list"] = ','.join([source.sensor_id for source in source_config.sensor_list])
            source_properties["sensor-name-list"] = ','.join([source.sensor_name for source in source_config.sensor_list])
            if is_audio_batch or record_audio:
                source_properties["disable-audio"] = False
            if is_audio_batch:
                source_properties["mode"] = 1
            self._pipeline.add("nvmultiurisrcbin", source, source_properties)
            streams.append(StreamInfo(source, media_type=batch_media_type))
            self._batch_size = source_properties["max-batch-size"]
        else:
            streammux_properties = {}
            for k, v in default_mux_properties.items():
                if is_audio_batch and k in {"width", "height"}:
                    continue
                if k in properties:
                    # override the source config properties
                    streammux_properties[k] = properties[k]
                else:
                    # use the default value
                    streammux_properties[k] = v
            mux = _get_node_name(func, 'mux')
            if not is_audio_batch:
                # Ensure GPU is used for transformations to avoid VIC scale factor issues
                streammux_properties["gpu-id"] = gpu_id
                streammux_properties["compute-hw"] = 1  # Use GPU instead of VIC
            self._pipeline.add("nvstreammux", mux, streammux_properties)
            if record_config and record_config.recording_type == "cloud":
                self._sr_controller = CommonFactory.create("smart_recording_action", "sr_controller")
                if isinstance(self._sr_controller, signal.Emitter):
                    sr_controller_properties = {
                        "proto-lib": record_config.proto_lib,
                        "conn-str": record_config.conn_str,
                        "msgconv-config-file": record_config.msgconv_config_file,
                        "proto-config-file": record_config.proto_config_file,
                        "topic-list": record_config.topic_list
                    }
                    self._sr_controller.set(sr_controller_properties)
                else:
                    raise ValueError("Failed to create smart recording controller")
            for i, uri in enumerate(uri_list):
                if record_config and record_config.recording_type == "cloud":
                    source_properties = {
                        "uri": _validate_uri(uri),
                        "gpu-id": gpu_id,
                        "smart-record": 1,
                        "smart-rec-cache": record_config.rec_cache,
                        "smart-rec-container": record_config.rec_container,
                        "smart-rec-dir-path": record_config.rec_dir_path,
                        "smart-rec-mode": record_config.rec_mode,
                    }
                elif record_config and record_config.recording_type == "local":
                    source_properties = {
                        "uri": _validate_uri(uri),
                        "gpu-id": gpu_id,
                        "smart-record": 2,
                        "smart-rec-cache": record_config.rec_cache,
                        "smart-rec-container": record_config.rec_container,
                        "smart-rec-dir-path": record_config.rec_dir_path,
                        "smart-rec-mode": record_config.rec_mode,
                    }
                else:
                    source_properties = {
                        "uri": _validate_uri(uri),
                        "gpu-id": gpu_id
                    }
                if is_audio_batch or record_audio:
                    source_properties["disable-audio"] = False
                # For cloud smart recording, use src_N names so the C++ plugin can parse id with std::stoll
                if source_config and record_config and record_config.recording_type == "cloud":
                    source_name = f"src_{i}"
                else:
                    source_name = source_config.sensor_list[i].sensor_id if source_config else f"{source}_{i}"
                source_type = source_config.source_type if source_config and source_config.source_type else self.DEFAULT_SOURCE_TYPE
                for k, v in default_source_properties.items():
                    if k in properties:
                        # override the source config properties
                        source_properties[k] = properties[k]
                    elif source_config and k in source_config.source_properties:
                        # use the source config properties
                        source_properties[k] = source_config.source_properties[k]
                    else:
                        # use the default value
                        source_properties[k] = v
                self._pipeline.add(source_type, source_name, source_properties)
                if is_audio_batch:
                    queue_name = _get_node_name(func, "audio-queue")
                    converter_name = _get_node_name(func, "audio-convert")
                    resample_name = _get_node_name(func, "audio-resample")
                    self._pipeline.add("queue", queue_name)
                    self._pipeline.add("audioconvert", converter_name)
                    self._pipeline.add("audioresample", resample_name)
                    self._pipeline.link((source_name, queue_name), ("asrc_%u", ""))
                    self._pipeline.link(queue_name, converter_name, resample_name)
                    self._pipeline.link((resample_name, mux), ("", "sink_%u"))
                else:
                    self._pipeline.link((source_name, mux), ("vsrc_%u", ""))
                if record_config and record_config.recording_type == "cloud":
                    self._sr_controller.attach("start-sr", self._pipeline[source_name])
                    self._sr_controller.attach("stop-sr", self._pipeline[source_name])
                    self._pipeline.attach(source_name, "smart_recording_signal", "sr", "sr-done")
                    
            streams.append(StreamInfo(mux, media_type=batch_media_type))
            self._batch_size = len(uri_list)

        return Flow(self._pipeline, streams=streams, parent=self)

    def inject(self, providers: list, **kwargs):
        """
        Inject raw data to the pipeline by creating buffers
        Supported format:
            raw:     ["RGB", "RGBA", "I420"]
            audio:   GStreamer raw audio formats such as "S16LE" or "F32LE"
            encoded: ["JPEG"]
        Args:
            provider(list(BufferProvider)): list of providers, width/height/format/framerate/device must be supplied for video.
                                           Audio providers must supply media_type="audio", format, sample_rate or rate,
                                           and channels or num_channels.
        Return: A derived flow
        Raises: TypeError
        """
        if not providers:
            raise TypeError("List of providers must be supplied for injecting data")
        streams = []
        for provider in providers:
            provider_format = provider.format.upper()
            if provider_format not in _mime_table:
                raise Exception(f"Format {provider.format} not supported")
            mime = _mime_table[provider_format]
            media_type = getattr(provider, "media_type", None)
            if media_type is None:
                media_type = "audio" if mime.startswith("audio/") else "video"
            media_type = media_type.lower()
            if media_type not in ["audio", "video"]:
                raise TypeError(f"Unsupported media type: {media_type}")
            if media_type == "video" and mime.startswith("audio/"):
                raise TypeError(f"Format {provider.format} is an audio format")
            if media_type == "audio" and not mime.startswith("audio/"):
                raise TypeError(f"Format {provider.format} is not an audio format")
            raw_video_data = provider_format in _raw_video_formats
            raw_audio_data = provider_format in _raw_audio_formats
            if mime == "video/x-raw" and hasattr(provider, "device") and 'gpu' in provider.device:
                mime += "(memory:NVMM)"
            source = _get_node_name(Flow.inject, "source")
            converter = _get_node_name(Flow.inject, "converter")
            capsfilter = _get_node_name(Flow.inject, "capsfilter")
            feeder = _get_node_name(Flow.inject, "feeder")
            if media_type == "audio":
                if not raw_audio_data:
                    raise TypeError(f"Format {provider.format} is not a raw audio format")
                sample_rate = getattr(provider, "sample_rate", getattr(provider, "rate", None))
                channels = getattr(provider, "channels", getattr(provider, "num_channels", None))
                if sample_rate is None or channels is None:
                    raise TypeError("Audio providers must supply sample_rate/rate and channels/num_channels")
                layout = getattr(provider, "layout", "interleaved")
                caps1 = getattr(
                    provider,
                    "caps",
                    f"{mime}, format={provider_format}, rate={sample_rate}, channels={channels}, layout={layout}"
                )
            else:
                caps1 = f"{mime}, format={provider_format}, width={provider.width}, height={provider.height}, framerate={provider.framerate}/1"
                caps2 = f"video/x-raw(memory:NVMM), format=NV12, width={provider.width}, height={provider.height}, framerate={provider.framerate}/1"
            self._pipeline.add("appsrc", source, {
                "caps": caps1,
                "do-timestamp": True
            }).attach(source, Feeder(feeder, provider), tips="need-data/enough-data")
            if media_type == "audio":
                streams.append(StreamInfo(source, media_type="audio"))
            elif raw_video_data:
                self._pipeline.add("nvvideoconvert", converter,  {"gpu-id": kwargs.get("gpu_id", 0), "compute-hw": 1})
                self._pipeline.add("capsfilter", capsfilter, {"caps": caps2})
                self._pipeline.link(source, converter, capsfilter)
                streams.append(StreamInfo(capsfilter, media_type="video"))
            else:
                streams.append(StreamInfo(source, media_type="video"))

        return Flow(self._pipeline, streams=streams, parent=self)

    def select(self, media_type: Optional[str] = None, originator: Optional[str] = None):
        """
        Select a subset of streams from the current flow.

        Args:
            media_type(str): optional "audio" or "video" stream filter
            originator(str): optional source element name filter
        Return: A derived flow
        Raises: Exception
        """
        streams = list(self._streams)
        if media_type:
            media_types = _normalize_media_types(media_type)
            if len(media_types) != 1:
                raise TypeError("select() accepts exactly one media type filter")
            streams = [stream for stream in streams if stream.media_type == media_types[0]]
        if originator:
            streams = [stream for stream in streams if stream.originator == originator]
        if not streams:
            raise Exception("No stream matched the selection")
        return Flow(self._pipeline, streams=streams, parent=self)


    def retrieve(self, retriever: BufferRetriever, **kwargs):
        """
        Retrieve the buffer from the pipeline.
        If users want to reuse the data in the buffer, it must be
        copied out from the callback, otherwise it'll be disposed
        by the pipeline.
        Args:
            retriever(BufferRetriever): retriever object to pull the data
        Return: A derived flow
        Raises: Upstream Exception
        """
        if len(self._streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")
        stream_info = _parse_stream_info(self._streams[0])
        if stream_info.media_type == "audio":
            queue = _get_node_name(Flow.retrieve, "audio-queue")
            converter = _get_node_name(Flow.retrieve, "audio-convert")
            resample = _get_node_name(Flow.retrieve, "audio-resample")
            sink = _get_node_name(Flow.retrieve, "appsink")
            receiver = _get_node_name(Flow.retrieve, "receiver")
            sink_properties = {"emit-signals": True, "sync": False}
            audio_caps = kwargs.pop("audio_caps", None)
            kwargs.pop("gpu_id", None)
            _populate_kwargs(sink_properties, **kwargs)
            self._pipeline.add("queue", queue)
            self._pipeline.add("audioconvert", converter)
            self._pipeline.add("audioresample", resample)
            if audio_caps:
                capsfilter = _get_node_name(Flow.retrieve, "audio-capsfilter")
                self._pipeline.add("capsfilter", capsfilter, {"caps": audio_caps})
                self._pipeline.add("appsink", sink, sink_properties).attach(
                    sink, Receiver(receiver, retriever), tips="new-sample"
                )
                self._pipeline.link(queue, converter, resample, capsfilter, sink)
            else:
                self._pipeline.add("appsink", sink, sink_properties).attach(
                    sink, Receiver(receiver, retriever), tips="new-sample"
                )
                self._pipeline.link(queue, converter, resample, sink)
            _link_stream(self._pipeline, stream_info, queue)
            return Flow(self._pipeline, parent=self)
        converter = _get_node_name(Flow.retrieve, "converter")
        capsfilter = _get_node_name(Flow.retrieve, "capsfilter")
        sink = _get_node_name(Flow.retrieve, "appsink")
        receiver = _get_node_name(Flow.retrieve, "receiver")
        gpu_id = kwargs.pop("gpu_id", 0)
        sink_properties = {"emit-signals": True, "sync": False}
        _populate_kwargs(sink_properties, **kwargs)
        self._pipeline.add("nvvideoconvert", converter, {"gpu-id": gpu_id, "compute-hw": 1})
        self._pipeline.add("appsink", sink, sink_properties).attach(sink, Receiver(receiver, retriever), tips="new-sample")
        self._pipeline.add("capsfilter", capsfilter, {"caps": "video/x-raw(memory:NVMM), format=RGB"})
        self._pipeline.link(converter, capsfilter, sink)
        if stream_info.template:
             self._pipeline.link(
                 (stream_info.originator, converter),
                 (stream_info.template, '')
             )
        else:
            self._pipeline.link(stream_info.originator, converter)
        return Flow(self._pipeline, parent=self)

    def batch(self, **kwargs):
        """
        Create a batch for buffers received from each upstream.
        By default, the batch size will be set to the number of upstreams
        Args:
            kwargs: overriding streammux properties(use in caution)
        Return: A derived flow
        Raises: Upstream Exception
        """
        #set the batch size
        if len(self._streams) == 0:
            raise Exception("Upstream error: no stream found")
        if self._batch_size != 0:
            raise Exception("Upstream error: already batched")

        parsed_streams = [_parse_stream_info(stream) for stream in self._streams]
        media_types = {stream_info.media_type for stream_info in parsed_streams}
        if len(media_types) != 1:
            raise Exception("Upstream error: audio and video streams cannot be batched together")
        batch_media_type = next(iter(media_types))
        is_audio_batch = batch_media_type == "audio"

        # populate the properties from kwargs
        properties = {}
        _populate_kwargs(properties, **kwargs)
        if "batch-size" not in properties:
            properties["batch-size"] = len(parsed_streams)
        if not is_audio_batch and "width" not in properties:
            properties["width"] = self.DEFAULT_PICTURE_WIDTH
        if not is_audio_batch and "height" not in properties:
            properties["height"] = self.DEFAULT_PICTURE_HEIGHT

        mux_name = _get_node_name(Flow.batch, "mux")
        if not is_audio_batch:
            # Ensure GPU is used for transformations to avoid VIC scale factor issues
            properties["gpu-id"] = properties.get("gpu-id", 0)
            properties["compute-hw"] = 1  # Use GPU instead of VIC
        self._pipeline.add("nvstreammux", mux_name, properties)

        for stream_info in parsed_streams:
            if is_audio_batch:
                queue_name = _get_node_name(Flow.batch, "audio-queue")
                converter_name = _get_node_name(Flow.batch, "audio-convert")
                resample_name = _get_node_name(Flow.batch, "audio-resample")
                self._pipeline.add("queue", queue_name)
                self._pipeline.add("audioconvert", converter_name)
                self._pipeline.add("audioresample", resample_name)
                _link_stream(self._pipeline, stream_info, queue_name)
                self._pipeline.link(queue_name, converter_name, resample_name)
                self._pipeline.link((resample_name, mux_name), ("", "sink_%u"))
            else:
                self._pipeline.link(
                    (stream_info.originator, mux_name),
                    (stream_info.template, "sink_%u")
                )

        self._batch_size = properties["batch-size"]
        return Flow(self._pipeline, streams=[StreamInfo(mux_name, media_type=batch_media_type)], parent=self)

    def decode(self, **kwargs):
        """
        Add decoders to the pipeline
        Return: A derived flow
        """
        streams = []
        for stream in self._streams:
            stream_info = _parse_stream_info(stream)
            decoder_name = _get_node_name(Flow.decode, "decode")
            properties = {}
            _populate_kwargs(properties, **kwargs)
            self._pipeline.add("decodebin", decoder_name, properties)
            if stream_info.media_type == "audio":
                converter_name = _get_node_name(Flow.decode, "audio-convert")
                resample_name = _get_node_name(Flow.decode, "audio-resample")
                _link_stream(self._pipeline, stream_info, decoder_name)
                self._pipeline.add("audioconvert", converter_name)
                self._pipeline.add("audioresample", resample_name)
                self._pipeline.link(decoder_name, converter_name, resample_name)
                streams.append(StreamInfo(resample_name, media_type="audio"))
            else:
                converter_name = _get_node_name(Flow.decode, "converter")
                gpu_id = properties["gpu-id"] if "gpu-id" in properties else 0
                _link_stream(self._pipeline, stream_info, decoder_name)
                self._pipeline.add("nvvideoconvert", converter_name, {"gpu-id": gpu_id, "compute-hw": 1})
                self._pipeline.link(decoder_name, converter_name)
                streams.append(StreamInfo(converter_name, media_type="video"))

        return Flow(self._pipeline, streams=streams, parent=self)

    def infer(self, config, with_triton=False, **kwargs):
        """
        Enable inference in the pipeline
        Args:
            config: the inference configuration
            kwargs: for overriding inference configuration(use in caution)
        Raises: Upstream Exception
        """
        if len(self._streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")

        stream_info = _parse_stream_info(self._streams[0])
        config_file_path = None
        if not isinstance(config, str):
            raise TypeError("Inference config must be a file path")
        elif os.path.exists(config):
            config_file_path = config
        infer_name = _get_node_name(Flow.infer, "infer")
        properties = {"config-file-path": config_file_path} if config_file_path else {}
        audio_caps = kwargs.pop("audio_caps", None)
        _populate_kwargs(properties, **kwargs)
        if stream_info.media_type == "audio":
            if with_triton:
                raise TypeError("Audio inference only supports nvinferaudio")
            element_name = "nvinferaudio"
            if self._batch_size:
                last_name = stream_info.originator
            else:
                convert_name = _get_node_name(Flow.infer, "audio-convert")
                resample_name = _get_node_name(Flow.infer, "audio-resample")
                self._pipeline.add("audioconvert", convert_name)
                self._pipeline.add("audioresample", resample_name)
                _link_stream(self._pipeline, stream_info, convert_name)
                self._pipeline.link(convert_name, resample_name)
                last_name = resample_name
            if audio_caps:
                capsfilter_name = _get_node_name(Flow.infer, "audio-caps")
                caps = _nvmm_audio_caps(audio_caps) if self._batch_size else audio_caps
                self._pipeline.add("capsfilter", capsfilter_name, {"caps": caps})
                if self._batch_size:
                    _link_stream(self._pipeline, stream_info, capsfilter_name)
                else:
                    self._pipeline.link(last_name, capsfilter_name)
                last_name = capsfilter_name
        else:
            element_name = "nvinferserver" if with_triton else "nvinfer"
            if "batch-size" not in properties:
                properties["batch-size"] = self._batch_size
            last_name = None
        self._pipeline.add(element_name, infer_name, properties)
        if stream_info.media_type == "audio":
            self._pipeline.link(last_name, infer_name)
        else:
            _link_stream(self._pipeline, stream_info, infer_name)
        return Flow(self._pipeline, streams=[StreamInfo(infer_name, media_type=stream_info.media_type)], parent=self)

    def analyze(self, config, **kwargs):
        """
        Enable analytics in the pipeline
        Args:
            kwargs: standard analytics properties
        Raises: Upstream Exception
        """
        if len(self._streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")
        stream_info = _parse_stream_info(self._streams[0])
        if stream_info.media_type == "audio":
            raise Exception("Upstream error: analyze is only supported for video streams")
        analytics = _get_node_name(Flow.analyze, "analyze")
        if not os.path.exists(config):
            raise FileNotFoundError(f"Config file {config} not found")
        properties = {"config-file": config}
        _populate_kwargs(properties, **kwargs)
        self._pipeline.add("nvdsanalytics", analytics, properties)
        _link_stream(self._pipeline, stream_info, analytics)
        return Flow(self._pipeline, streams=[StreamInfo(analytics, media_type="video")], parent=self)

    def track(self, **kwargs):
        """
        Track the detected object, must come after primary inference
        Args:
            kwargs: standard tracker properties
        Raises: Upstream Exception
        """
        if len(self._streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")
        stream_info = _parse_stream_info(self._streams[0])
        if stream_info.media_type == "audio":
            raise Exception("Upstream error: track is only supported for video streams")
        tracker = _get_node_name(Flow.track, "tracker")
        properties = {}
        _populate_kwargs(properties, **kwargs)
        self._pipeline.add("nvtrackerbin", tracker, properties)
        _link_stream(self._pipeline, stream_info, tracker)
        return Flow(self._pipeline, streams=[StreamInfo(tracker, media_type="video")], parent=self)

    def render(self, mode: RenderMode = RenderMode.DISPLAY, enable_osd = True, rtsp_mount_point: str = 'ds-test5', rtsp_port: int = 8554, seg_mask_config: dict = None, **kwargs):
        """
        Render the video buffers on a display
        Args:
            mode:   renderer mode, DISCARD for dropping output
            enable_osd: enable Overlay Display
            rtsp_mount_point: rtsp mount point
            rtsp_port: rtsp port
            seg_mask_config: a dictionary of segmentation mask configuration
            kwargs: standard sink parameter with kwargs
        Return: A derived flow
        Raises: Upstream Exception
        """
        if len(self._streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")

        properties = {}
        _populate_kwargs(properties, **kwargs)
        stream_info = _parse_stream_info(self._streams[0])
        sink_name = _get_node_name(Flow.render, "sink")
        if stream_info.media_type == "audio":
            if mode not in (RenderMode.DISPLAY, RenderMode.DISCARD, RenderMode.STREAM):
                raise ValueError("Invalid render mode")
            if mode == RenderMode.DISCARD:
                self._pipeline.add("fakesink", sink_name, properties)
                _link_stream(self._pipeline, stream_info, sink_name)
                return Flow(self._pipeline, parent=self)
            audio_sink = properties.pop("audio-sink", "autoaudiosink")
            queue_name = _get_node_name(Flow.render, "audio-queue")
            converter_name = _get_node_name(Flow.render, "audio-convert")
            resample_name = _get_node_name(Flow.render, "audio-resample")
            self._pipeline.add("queue", queue_name)
            self._pipeline.add("audioconvert", converter_name)
            self._pipeline.add("audioresample", resample_name)
            _link_stream(self._pipeline, stream_info, queue_name)
            self._pipeline.link(queue_name, converter_name, resample_name)
            if mode == RenderMode.STREAM:
                sink_type = "nvrtspoutsinkbin"
                sink_properties = {
                    "rtsp-port": rtsp_port,
                    "rtsp-mount-point": _normalize_rtsp_mount_point(rtsp_mount_point),
                    "sync": False if "sync" not in kwargs else kwargs["sync"],
                }
                sink_properties.update(properties)
                self._pipeline.add(sink_type, sink_name, sink_properties)
                self._pipeline.link((resample_name, sink_name), ("", "asink"))
            else:
                self._pipeline.add(audio_sink, sink_name, properties)
                self._pipeline.link(resample_name, sink_name)
        else:
            gpu_id = properties["gpu-id"] if "gpu-id" in properties else 0
            tiler_name = _get_node_name(Flow.render, "tiler")
            converter_name = _get_node_name(Flow.render, "convert")
            osd_name = _get_node_name(Flow.render, "osd")
            seg_visual_name = _get_node_name(Flow.render, "nvsegvisual")
            if mode == RenderMode.DISCARD:
                self._pipeline.add("fakesink", sink_name, properties)
                if stream_info.template:
                    self._pipeline.link(
                        (stream_info.originator, sink_name),
                        (stream_info.template, '')
                    )
                else:
                    self._pipeline.link(stream_info.originator, sink_name)
            elif mode in (RenderMode.DISPLAY, RenderMode.STREAM):
                sink_type = 'nv3dsink' if platform.processor() == 'aarch64' else 'nveglglessink'

                self._pipeline.add("nvvideoconvert", converter_name, {"gpu-id": gpu_id, "compute-hw": 1})
                if stream_info.template:
                    self._pipeline.link(
                        (stream_info.originator, converter_name),
                        (stream_info.template, '')
                    )
                else:
                    self._pipeline.link(stream_info.originator, converter_name)
                last_name = converter_name
                if seg_mask_config and isinstance(seg_mask_config, dict):
                    seg_visual_properties = {
                        "gpu-id": gpu_id,
                        "original-background": 1,
                        "alpha": 0.5
                    }
                    seg_visual_properties.update(seg_mask_config)
                    self._pipeline.add("nvsegvisual", seg_visual_name, seg_visual_properties)
                    self._pipeline.link(last_name, seg_visual_name)
                    last_name = seg_visual_name
                if self._batch_size > 1:
                    tiler_properties = {
                        "gpu-id": gpu_id,
                        "width": properties.pop("width", self.DEFAULT_PICTURE_WIDTH),
                        "height": properties.pop("height", self.DEFAULT_PICTURE_HEIGHT)
                    }
                    self._pipeline.add("nvmultistreamtiler", tiler_name, tiler_properties)
                    self._pipeline.link(last_name, tiler_name)
                    last_name = tiler_name
                if enable_osd:
                    self._pipeline.add("nvdsosd", osd_name, {"gpu-id": gpu_id, "display-mask": True})
                    self._pipeline.link(last_name, osd_name)
                    last_name = osd_name

                if mode == RenderMode.STREAM:
                    sink_type = 'nvrtspoutsinkbin'
                    properties = {
                        "rtsp-port": rtsp_port,
                        "rtsp-mount-point": _normalize_rtsp_mount_point(rtsp_mount_point),
                        "sync": False if "sync" not in kwargs else kwargs["sync"],
                    }

                self._pipeline.add(sink_type, sink_name, properties).link(last_name, sink_name)
            else:
                raise ValueError("Invalid render mode")
        return Flow(self._pipeline, parent=self)

    def attach(self, what, name="", tips="", properties=None):
        """
        Attach a probe to the current flow
        Args:
            what:       object or name of the module that creates an object
            name:       name of the object, not applicable for explicitly created object
            tips (str): extra information for the custom object
            properties (Dict): properties to be set on the object, not applicable for explicitly created object
        Return: A derived flow
        """
        for stream in self._streams:
            stream_info = _parse_stream_info(stream)
            self._pipeline.attach(stream_info.originator, what, name, tips or stream_info.template, properties)
        return Flow(self._pipeline, streams=self._streams, parent=self)

    def fork(self):
        """
        Fork a flow to support multiple processing branches
        Args: None
        Return: A derived flow
        """
        if len(self._streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")
        tee = _get_node_name(Flow.fork, "tee")
        stream_info = _parse_stream_info(self._streams[0])
        self._pipeline.add("tee", tee).link(
            (stream_info.originator, tee),
            (stream_info.template, "")
        )
        return Flow(self._pipeline, [StreamInfo(tee, media_type=stream_info.media_type)], parent=self)

    def publish(self, **kwargs):
        """
        Publish generated events to a remote server
        Args:
            kwargs: standard nvmsgconv/nvmsgbroker properties
        Return: A derived flow
        Raises: Upstream Exception
        """
        if len(self._streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")
        sink = _get_node_name(Flow.publish, "msgsink")
        stream_info = _parse_stream_info(self._streams[0])
        sink_properties = { "sync": False }
        _populate_kwargs(sink_properties, **kwargs)
        if stream_info.media_type == "audio":
            sink_properties.setdefault("msg-conv-msg2p-lib", _default_audio_msg2p_lib)
        self._pipeline.add("nvmsgbrokersinkbin", sink, sink_properties).link(
            (stream_info.originator, sink),
            (stream_info.template, "")
        )
        return Flow(self._pipeline, parent=self)

    def encode(self, dest: str, use_sw_codec=False, enable_osd=True, **kwargs):
        """
        Encode the stream to a file or through RTSP streaming
        Args:
            dest(str): support "file://" and "udp://"
            use_sw_codec: flag for switching to software codec
            enable_osd: enable Overlay Display
            kwargs: encoding options: profile/bitrate/iframeinterval/codec_type
                    sink options: sync
        Return: A derived flow
        Raises: Upstream Exception
        """
        uri = urlparse(dest)
        to_file = True if not uri.scheme or uri.scheme == "file" else False
        file_ext = os.path.splitext(uri.path)[1] if to_file else ""
        streams = [_parse_stream_info(stream) for stream in self._streams]
        audio_streams = [stream for stream in streams if stream.media_type == "audio"]
        video_streams = [stream for stream in streams if stream.media_type == "video"]

        if len(streams) == 2 and len(audio_streams) == 1 and len(video_streams) == 1:
            audio_stream = audio_streams[0]
            video_stream = video_streams[0]
            if audio_stream.originator != video_stream.originator:
                raise Exception("Upstream error: mixed audio/video encode requires streams from the same source")
            if not to_file:
                raise TypeError("Mixed audio/video encode currently supports file outputs only.")
            if file_ext.upper() not in [".MP4", ".MOV"]:
                raise TypeError(f"File extension {file_ext} is not supported for mixed audio/video outputs")

            audio_encoder_type = kwargs.pop("audio_encoder", None)
            audio_parser_type = kwargs.pop("audio_parser", None)
            audio_mux = kwargs.pop("audio_mux", None)
            audio_caps = kwargs.pop("audio_caps", None)
            if audio_mux:
                raise TypeError("audio_mux is not supported for mixed audio/video outputs")

            video_queue = _get_node_name(Flow.encode, "mux-video-queue")
            video_converter = _get_node_name(Flow.encode, "mux-video-convert")
            video_osd = _get_node_name(Flow.encode, "mux-video-osd")
            video_post_osd_converter = _get_node_name(Flow.encode, "mux-video-post-osd-convert")
            video_capsfilter = _get_node_name(Flow.encode, "mux-video-capsfilter")
            video_encoder = _get_node_name(Flow.encode, "mux-video-encoder")
            video_parser = _get_node_name(Flow.encode, "mux-video-parser")
            video_mux_queue = _get_node_name(Flow.encode, "mux-video-mux-queue")
            audio_queue = _get_node_name(Flow.encode, "mux-audio-queue")
            audio_converter = _get_node_name(Flow.encode, "mux-audio-convert")
            audio_resample = _get_node_name(Flow.encode, "mux-audio-resample")
            audio_capsfilter = _get_node_name(Flow.encode, "mux-audio-caps")
            audio_encoder = _get_node_name(Flow.encode, "mux-audio-encoder")
            audio_parser = _get_node_name(Flow.encode, "mux-audio-parser")
            audio_mux_queue = _get_node_name(Flow.encode, "mux-audio-mux-queue")
            mux = _get_node_name(Flow.encode, "mux")
            sink = _get_node_name(Flow.encode, "filesink")

            mem_type = "video/x-raw" if use_sw_codec else "video/x-raw(memory:NVMM)"
            capsfilter_properties = { "caps": f"{mem_type}, format=I420"}
            encoder_properties = {
                "profile":  kwargs["profile"] if "profile" in kwargs else 0,
                "iframeinterval":  kwargs["iframeinterval"] if "iframeinterval" in kwargs else 10,
                "bitrate": kwargs["bitrate"] if "bitrate" in kwargs else 2000000,
            }
            gpu_id = kwargs["gpu_id"] if "gpu_id" in kwargs else 0
            converter_properties = {"gpu-id": gpu_id, "compute-hw": 1}
            osd_properties = {"gpu-id": gpu_id, "display-mask": True}
            queue_properties = {"max-size-bytes": 0, "max-size-time": 0, "max-size-buffers": 0}
            file_sink_properties = {
                "location": uri.path,
                "sync": False if "sync" not in kwargs else kwargs["sync"],
                "async": False
            }

            self._pipeline.add("queue", video_queue).add("nvvideoconvert", video_converter, converter_properties)
            _link_stream(self._pipeline, video_stream, video_queue)
            self._pipeline.link(video_queue, video_converter)
            video_last = video_converter
            if enable_osd:
                self._pipeline.add("nvdsosd", video_osd, osd_properties)
                self._pipeline.link(video_last, video_osd)
                video_last = video_osd
                self._pipeline.add("nvvideoconvert", video_post_osd_converter, converter_properties)
                self._pipeline.link(video_last, video_post_osd_converter)
                video_last = video_post_osd_converter
            self._pipeline.add("capsfilter", video_capsfilter, capsfilter_properties)
            self._pipeline.link(video_last, video_capsfilter)

            codec_type = kwargs.get("codec_type", "h264").lower()
            match codec_type:
                case "h264":
                    codec = "x264enc" if use_sw_codec else "nvv4l2h264enc"
                    self._pipeline.add(codec, video_encoder, encoder_properties).add("h264parse", video_parser)
                case "h265":
                    codec = "x265enc" if use_sw_codec else "nvv4l2h265enc"
                    self._pipeline.add(codec, video_encoder, encoder_properties).add("h265parse", video_parser)
                case "av1":
                    codec = "av1enc" if use_sw_codec else "nvv4l2av1enc"
                    self._pipeline.add(codec, video_encoder, encoder_properties).add("av1parse", video_parser)
                case _:
                    raise TypeError(f"Codec type {codec_type} is not supported")
            self._pipeline.link(video_capsfilter, video_encoder, video_parser)
            self._pipeline.add("queue", video_mux_queue, queue_properties)
            self._pipeline.link(video_parser, video_mux_queue)

            self._pipeline.add("queue", audio_queue).add("audioconvert", audio_converter).add("audioresample", audio_resample)
            _link_stream(self._pipeline, audio_stream, audio_queue)
            self._pipeline.link(audio_queue, audio_converter, audio_resample)
            audio_last = audio_resample
            if audio_caps is None:
                audio_caps = "audio/x-raw, format=S16LE, layout=interleaved, rate=48000"
            if audio_caps:
                self._pipeline.add("capsfilter", audio_capsfilter, {"caps": audio_caps})
                self._pipeline.link(audio_last, audio_capsfilter)
                audio_last = audio_capsfilter
            # The validation image ships mp4/qt muxers but not AAC encoders, so
            # prefer Opus for MP4 and raw PCM for MOV unless the caller overrides it.
            selected_audio_encoder = None
            if audio_encoder_type is None and file_ext.upper() == ".MOV":
                if audio_parser_type:
                    raise TypeError("audio_parser requires audio_encoder for raw-audio .mov outputs")
            else:
                if audio_encoder_type is None:
                    encoder_candidates = ("opusenc", "voaacenc", "avenc_aac", "faac", "fdkaacenc")
                    selected_audio_encoder = _add_first_available_element(
                        self._pipeline, encoder_candidates, audio_encoder
                    )
                else:
                    self._pipeline.add(audio_encoder_type, audio_encoder)
                    selected_audio_encoder = audio_encoder_type
                self._pipeline.link(audio_last, audio_encoder)
                audio_last = audio_encoder
                selected_audio_parser = (
                    audio_parser_type
                    if audio_parser_type is not None
                    else _default_audio_parser_for_encoder(selected_audio_encoder)
                )
                if selected_audio_parser:
                    self._pipeline.add(selected_audio_parser, audio_parser)
                    self._pipeline.link(audio_last, audio_parser)
                    audio_last = audio_parser
            self._pipeline.add("queue", audio_mux_queue, queue_properties)
            self._pipeline.link(audio_last, audio_mux_queue)

            mux_type = "qtmux" if file_ext.upper() == ".MOV" else "mp4mux"
            self._pipeline.add(mux_type, mux).add("filesink", sink, file_sink_properties)
            self._pipeline.link((video_mux_queue, mux), ("", "video_%u"))
            self._pipeline.link((audio_mux_queue, mux), ("", "audio_%u"))
            self._pipeline.link(mux, sink)
            return Flow(self._pipeline, parent=self)

        if len(streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")

        stream_info = streams[0]
        if stream_info.media_type == "audio":
            if not to_file:
                raise TypeError("Audio encode currently supports file outputs only.")
            audio_encoder = kwargs.pop("audio_encoder", None)
            audio_parser = kwargs.pop("audio_parser", None)
            audio_mux = kwargs.pop("audio_mux", None)
            audio_caps = kwargs.pop("audio_caps", None)
            sink = _get_node_name(Flow.encode, "audio-filesink")
            queue = _get_node_name(Flow.encode, "audio-queue")
            converter = _get_node_name(Flow.encode, "audio-convert")
            resample = _get_node_name(Flow.encode, "audio-resample")
            file_sink_properties = {
                "location": uri.path,
                "sync": False if "sync" not in kwargs else kwargs["sync"],
                "async": False,
            }
            self._pipeline.add("queue", queue)
            self._pipeline.add("audioconvert", converter)
            self._pipeline.add("audioresample", resample)
            _link_stream(self._pipeline, stream_info, queue)
            self._pipeline.link(queue, converter, resample)
            last_name = resample
            if audio_caps:
                capsfilter = _get_node_name(Flow.encode, "audio-caps")
                self._pipeline.add("capsfilter", capsfilter, {"caps": audio_caps})
                self._pipeline.link(last_name, capsfilter)
                last_name = capsfilter
            if file_ext.upper() == ".WAV" and not audio_encoder and not audio_parser and not audio_mux:
                encoder = _get_node_name(Flow.encode, "audio-wavenc")
                self._pipeline.add("wavenc", encoder)
                self._pipeline.add("filesink", sink, file_sink_properties)
                self._pipeline.link(last_name, encoder, sink)
            else:
                if not audio_encoder:
                    raise TypeError("Custom audio outputs require audio_encoder or a .wav destination")
                encoder = _get_node_name(Flow.encode, "audio-encoder")
                self._pipeline.add(audio_encoder, encoder)
                self._pipeline.link(last_name, encoder)
                last_name = encoder
                if audio_parser:
                    parser = _get_node_name(Flow.encode, "audio-parser")
                    self._pipeline.add(audio_parser, parser)
                    self._pipeline.link(last_name, parser)
                    last_name = parser
                if audio_mux:
                    mux = _get_node_name(Flow.encode, "audio-mux")
                    self._pipeline.add(audio_mux, mux)
                    self._pipeline.link(last_name, mux)
                    last_name = mux
                self._pipeline.add("filesink", sink, file_sink_properties)
                self._pipeline.link(last_name, sink)
            return Flow(self._pipeline, parent=self)

        format_list = ['.MP4', '.MOV', '.JPG', '.MJPG']
        if not to_file and uri.scheme != "udp" and uri.scheme != "rtsp":
            raise TypeError("Only file, rtsp or udp are supported.")
        if to_file and file_ext.upper() not in format_list:
            raise TypeError(f"File extension {file_ext} is not supported")
        if not to_file and uri.hostname and not uri.hostname in ["localhost", "127.0.0.1"]:
            raise TypeError("Only localhost is supported for streaming")

        queue = _get_node_name(Flow.encode, "queue")
        converter = _get_node_name(Flow.encode, "convert")
        osd = _get_node_name(Flow.encode, "osd")
        post_osd_converter = _get_node_name(Flow.encode, "post-osd-convert")
        capsfilter = _get_node_name(Flow.encode, "capsfilter")
        encoder = _get_node_name(Flow.encode, "encoder")
        parser = _get_node_name(Flow.encode, "parser")
        mux = _get_node_name(Flow.encode, "mux")
        sink = _get_node_name(Flow.encode, "filesink")
        mem_type = "video/x-raw" if use_sw_codec else "video/x-raw(memory:NVMM)"
        capsfilter_properties = { "caps": f"{mem_type}, format=I420"}
        encoder_properties = {
            "profile":  kwargs["profile"] if "profile" in kwargs else 0,
            "iframeinterval":  kwargs["iframeinterval"] if "iframeinterval" in kwargs else 10,
            "bitrate": kwargs["bitrate"] if "bitrate" in kwargs else 2000000,
        }
        gpu_id = kwargs["gpu_id"] if "gpu_id" in kwargs else 0
        converter_properties = {"gpu-id": gpu_id, "compute-hw": 1}
        osd_properties = {"gpu-id": gpu_id, "display-mask": True}
        file_sink_properties = { "location": uri.path, "sync": False if "sync" not in kwargs else kwargs["sync"], "async": False }
        rtsp_sink_properties = { "rtsp-port": uri.port, "sync": False if "sync" not in kwargs else kwargs["sync"], "async": False }
        udp_sink_properties = { "host": "127.0.0.1", "port": uri.port, "sync": False if "sync" not in kwargs else kwargs["sync"], "async": False }

        def _add_encode_head(add_post_osd_converter=False):
            self._pipeline.add("queue", queue).add("nvvideoconvert", converter, converter_properties)
            link_nodes = [queue, converter]
            if enable_osd:
                self._pipeline.add("nvdsosd", osd, osd_properties)
                link_nodes.append(osd)
                if add_post_osd_converter:
                    self._pipeline.add("nvvideoconvert", post_osd_converter, converter_properties)
                    link_nodes.append(post_osd_converter)
            self._pipeline.link(
                (stream_info.originator, queue),
                (stream_info.template, "")
            ).link(*link_nodes)
            return link_nodes[-1]

        if uri.scheme == "rtsp":
            last_name = _add_encode_head()
            self._pipeline.add("nvrtspoutsinkbin", sink, rtsp_sink_properties).link(last_name, sink)
        elif file_ext.upper() in ['.JPG', '.MJPG']:
            # JPEG encoder:
            codec = "jpegenc" if use_sw_codec else "nvjpegenc"
            last_name = _add_encode_head(add_post_osd_converter=True)
            self._pipeline.add("capsfilter", capsfilter, capsfilter_properties)
            self._pipeline.add(codec, encoder, encoder_properties)
            self._pipeline.add("filesink", sink, file_sink_properties)
            self._pipeline.link(last_name, capsfilter, encoder, sink)
        else:
            last_name = _add_encode_head(add_post_osd_converter=True)
            self._pipeline.add("capsfilter", capsfilter, capsfilter_properties)
            codec_type = kwargs.get("codec_type", "h264").lower()
            match codec_type:
                case "h264":
                    codec = "x264enc" if use_sw_codec else "nvv4l2h264enc"
                    self._pipeline.add(codec, encoder, encoder_properties).add("h264parse", parser)
                case "h265":
                    codec = "x265enc" if use_sw_codec else "nvv4l2h265enc"
                    self._pipeline.add(codec, encoder, encoder_properties).add("h265parse", parser)
                case "av1":
                    codec = "av1enc" if use_sw_codec else "nvv4l2av1enc"
                    self._pipeline.add(codec, encoder, encoder_properties).add("av1parse", parser)
                case _:
                    raise TypeError(f"Codec type {codec_type} is not supported")
            if to_file:
                self._pipeline.add("mp4mux", mux).add("filesink", sink, file_sink_properties)
            else:
                self._pipeline[parser].set({"config-interval": -1})
                match codec_type:
                    case "h264":
                        self._pipeline.add("rtph264pay", mux)
                    case "h265":
                        self._pipeline.add("rtph265pay", mux)
                    case "av1":
                        # rtpav1pay and rtpav1depay plugins provided by gst-plugins-rs
                        self._pipeline.add("rtpav1pay", mux)
                self._pipeline.add("udpsink", sink, udp_sink_properties)

            self._pipeline.link(last_name, capsfilter, encoder, parser, mux, sink)

        return Flow(self._pipeline, parent=self)

    def preprocess(self, config_file: str, tensor_generator: Callable[[None], dict[str, Tensor]]=None, **kwargs):
        """
        Preprocess the stream
        """
        class TensorGenerator(BatchMetadataOperator):
            def __init__(self, tensor_generator: Callable[[int], dict[str, Tensor]]=None, target_unique_id: int=1):
                super().__init__()
                self._tensor_generator = tensor_generator
                self._target_unique_id = target_unique_id

            def handle_metadata(self, batch_meta):
                tensors = self._tensor_generator(batch_meta.n_frames)
                for name, tensor in tensors.items():
                    tensor_batch = batch_meta.acquire_preprocess_batch_meta()
                    tensor_batch.set_preprocessed_tensor(name, 0, self._target_unique_id, tensor)
                    batch_meta.append(tensor_batch)

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found")
        stream_info = _parse_stream_info(self._streams[0])
        if stream_info.media_type == "audio":
            raise Exception("Upstream error: preprocess is only supported for video streams")
        properties = { 'config-file': config_file, 'target-unique-ids': 1 }
        _populate_kwargs(properties, **kwargs)
        preprocess_name = _get_node_name(Flow.preprocess, "preprocess")
        queue_name = _get_node_name(Flow.preprocess, "queue")
        self._pipeline.add("nvdspreprocess", preprocess_name, properties).add("queue", queue_name)
        if tensor_generator:
            probe = Probe("probe", TensorGenerator(tensor_generator, properties["target-unique-ids"]))
            self._pipeline.attach(preprocess_name, probe)
        _link_stream(self._pipeline, stream_info, preprocess_name)
        self._pipeline.link(preprocess_name, queue_name)
        return Flow(self._pipeline, streams=[StreamInfo(queue_name, media_type="video")], parent=self)

    def apply(self, template_config: str = None, template_lib: str = None, **kwargs):
        """
        Apply custom video transformation using nvdsvideotemplate plugin

        Args:
            template_config (str): Path to configuration file (optional)
            template_lib (str): Name of custom library to load (optional)
            **kwargs: Additional properties for nvdsvideotemplate

        Returns:
            Flow: A new flow with custom transformation applied
        """
        if len(self._streams) != 1:
            raise Exception("Upstream error: multiple streams must be batched")
        stream_info = _parse_stream_info(self._streams[0])
        if stream_info.media_type == "audio":
            raise Exception("Upstream error: apply is only supported for video streams")

        template_name = _get_node_name(Flow.apply, "apply")
        properties = {}

        # Set config file if provided
        if template_config:
            if not os.path.exists(template_config):
                raise FileNotFoundError(f"Config file {template_config} not found")
            properties["config-file"] = template_config

        # Set custom library name if provided
        if template_lib:
            properties["customlib-name"] = template_lib

        # Set GPU ID if not specified
        if "gpu-id" not in properties and "gpu_id" not in kwargs:
            properties["gpu-id"] = kwargs.get("gpu_id", 0)

        # Set other properties
        _populate_kwargs(properties, **kwargs)

        # Add the nvdsvideotemplate element to pipeline
        self._pipeline.add("nvdsvideotemplate", template_name, properties)

        # Link the element
        _link_stream(self._pipeline, stream_info, template_name)

        return Flow(self._pipeline, streams=[StreamInfo(template_name, media_type="video")], parent=self)
