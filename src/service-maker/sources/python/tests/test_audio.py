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

from pyservicemaker import (
    Pipeline,
    Flow,
    MediaType,
    RenderMode,
    BufferRetriever,
    BufferProvider,
    Buffer,
    BatchMetadataOperator,
    Probe,
    RecordConfig,
)
from pyservicemaker.utils import MediaInfo, AudioStreamInfo, VideoStreamInfo
import os
import pytest
import threading
import time
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[3]

def _first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]

def _infer_config_asset_path(config_path, key):
    config_dir = Path(config_path).parent
    with open(config_path, encoding="utf-8") as config_file:
        for line in config_file:
            line = line.strip()
            if line.startswith(f"{key}="):
                return (config_dir / line.split("=", 1)[1]).resolve()
    return None

AUDIO_FILE = "/opt/nvidia/deepstream/deepstream/samples/streams/sonyc_mixed_audio.wav"
AV_SOURCE_FILE = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"
OUTPUT_FILE = "/tmp/pysm-audio-test.wav"
AV_OUTPUT_FILE = "/tmp/pysm-audio-video-test.mp4"
SMART_RECORD_RTSP_URI_ENV = "PYSM_AUDIO_SMART_RECORD_RTSP_URI"
MSG_BROKER_PROTO_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
DEFAULT_AUDIO_MSG2P_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_msgconv_audio.so"
SONYC_INFER_CONFIG = _first_existing_path(
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-audio/configs/config_infer_audio_sonyc.txt",
    str(SDK_ROOT / "apps/deepstream/sample_apps/deepstream-audio/configs/config_infer_audio_sonyc.txt"),
)
SONYC_AUDIO_TRANSFORM = (
    "melsdb,fft_length=2560,hop_size=692,dsp_window=hann,num_mels=128,"
    "sample_rate=44100,p2db_ref=(float)1.0,p2db_min_power=(float)0.0,"
    "p2db_top_db=(float)80.0"
)
SONYC_AUDIO_RATE = 44100
SONYC_AUDIO_FRAME_SIZE = 441000
SONYC_AUDIO_HOP_SIZE = 110250

class AudioBufferRetriever(BufferRetriever):
    def __init__(self, min_buffers=1):
        super().__init__()
        self.min_buffers = min_buffers
        self.buffers = 0
        self.done = threading.Event()

    def consume(self, buffer):
        self.buffers += 1
        if self.buffers >= self.min_buffers:
            self.done.set()
        return 1

class AudioBufferProvider(BufferProvider):
    def __init__(self):
        super().__init__()
        self.media_type = "audio"
        self.format = "S16LE"
        self.sample_rate = 16000
        self.channels = 1
        self.layout = "interleaved"
        self.buffers = 0
        self.expected_buffers = 8

    def generate(self, size):
        if self.buffers >= self.expected_buffers:
            return Buffer()
        self.buffers += 1
        # 20 ms of S16LE mono silence.
        return Buffer([0] * (self.sample_rate // 50 * self.channels * 2))

class AudioMetadataValidator(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.audio_batches = 0
        self.audio_frames = 0
        self.extracted_batches = 0
        self.acquire_frame_meta_rejected = False
        self.event_messages_generated = 0
        self.failure = None
        self.done = threading.Event()

    def handle_metadata(self, batch_meta):
        try:
            self.audio_batches += 1
            assert batch_meta.is_audio
            assert list(batch_meta.frame_items) == []

            try:
                batch_meta.acquire_frame_meta()
            except TypeError:
                self.acquire_frame_meta_rejected = True

            extracted = batch_meta.extract(max(1, batch_meta.n_frames))
            assert isinstance(extracted, list)
            assert len(extracted) == max(1, batch_meta.n_frames)
            for metadata in extracted:
                if metadata is None:
                    continue
                assert metadata["media_type"] == "audio"
                assert metadata["sample_rate"] > 0
                assert metadata["num_channels"] > 0
                assert metadata["num_samples_per_frame"] > 0
                assert isinstance(metadata["labels"], list)
                self.extracted_batches += 1

            for audio_frame_meta in batch_meta.audio_frame_items:
                assert audio_frame_meta.sample_rate > 0
                assert audio_frame_meta.num_channels > 0
                assert audio_frame_meta.num_samples_per_frame > 0
                assert audio_frame_meta.source_id >= 0
                assert audio_frame_meta.batch_id >= 0
                assert isinstance(audio_frame_meta.infer_done, bool)
                assert isinstance(audio_frame_meta.class_label, str)
                assert isinstance(list(audio_frame_meta.classifier_items), list)
                event_meta = batch_meta.acquire_event_message_meta()
                event_meta.generate(audio_frame_meta, sensor="test-sensor", uri=AUDIO_FILE)
                audio_frame_meta.append(event_meta)
                assert len(list(audio_frame_meta.user_meta_items(0))) >= 0
                self.audio_frames += 1
                self.event_messages_generated += 1
            self.done.set()
        except Exception as exc:
            self.failure = exc
            self.done.set()
            raise

class AudioBatchValidator(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.audio_batches = 0
        self.audio_frames = 0
        self.failure = None
        self.done = threading.Event()

    def handle_metadata(self, batch_meta):
        try:
            self.audio_batches += 1
            assert batch_meta.is_audio
            assert list(batch_meta.frame_items) == []
            frames = list(batch_meta.audio_frame_items)
            assert len(frames) > 0
            assert batch_meta.n_frames == len(frames)
            self.audio_frames += len(frames)
            self.done.set()
        except Exception as exc:
            self.failure = exc
            self.done.set()
            raise

def stop_pipeline(pipeline, timeout=10):
    time.sleep(timeout)
    pipeline.stop()

def stop_pipeline_on_event(pipeline, event, timeout=10):
    if not event.wait(timeout):
        event.timed_out = True
    pipeline.stop()

def _find_cached_node_name(pipeline, prefix):
    for name in pipeline._nodes:
        if name.startswith(prefix):
            return name
    raise AssertionError(f"Unable to find node with prefix {prefix}")

class InspectablePipeline(Pipeline):
    def __init__(self, name):
        super().__init__(name)
        self.added_properties = {}

    def add(self, type_name, name, properties=None):
        self.added_properties[name] = dict(properties or {})
        return super().add(type_name, name, properties)

def test_audio_smart_record_modes_enable_source_audio(tmp_path):
    for rec_mode in (0, 2):
        pipeline = InspectablePipeline(f"test-audio-smart-record-{rec_mode}")
        flow = Flow(pipeline).batch_capture(
            [AV_SOURCE_FILE],
            record_config=RecordConfig(
                recording_type="local",
                rec_cache=5,
                rec_container=0,
                rec_dir_path=str(tmp_path),
                rec_mode=rec_mode,
            ),
        )
        source = _find_cached_node_name(flow.pipeline, "batch_capture-source-")
        assert pipeline.added_properties[source]["smart-record"] == 2
        assert pipeline.added_properties[source]["smart-rec-mode"] == rec_mode
        assert flow.pipeline[source].get("disable-audio") is False

@pytest.mark.parametrize(
    "rec_mode, expected_video, expected_audio",
    [
        (0, True, True),
        (1, True, False),
        (2, False, True),
    ],
)
def test_audio_smart_record_rtsp_callback(tmp_path, rec_mode, expected_video, expected_audio):
    rtsp_uri = os.environ.get(SMART_RECORD_RTSP_URI_ENV)
    if not rtsp_uri:
        pytest.skip(f"Set {SMART_RECORD_RTSP_URI_ENV} to an audio/video RTSP URI")

    pipeline = Pipeline(f"test-audio-smart-record-callback-{rec_mode}")
    pipeline.add(
        "nvurisrcbin",
        "src_0",
        {
            "uri": rtsp_uri,
            "smart-record": 2,
            "smart-rec-cache": 5,
            "smart-rec-container": 0,
            "smart-rec-dir-path": str(tmp_path),
            "smart-rec-mode": rec_mode,
            "disable-audio": False,
        },
    ).add(
        "fakesink",
        "sink",
        {"sync": False},
    ).link(
        ("src_0", "sink"),
        ("vsrc_%u", ""),
    )

    done = threading.Event()
    result = {}
    failures = []

    def on_recording_done(info):
        result.update(
            {
                "file_path": os.path.join(info.file_directory, info.file_name),
                "duration": info.duration,
                "container_type": info.container_type,
                "contains_video": info.contains_video,
                "contains_audio": info.contains_audio,
                "width": info.width,
                "height": info.height,
                "channels": info.channels,
                "sampling_rate": info.sampling_rate,
            }
        )
        done.set()

    def record_and_stop():
        try:
            time.sleep(5)
            pipeline.start_recording("src_0", 0, 3, on_recording_done)
            if not done.wait(20):
                failures.append(AssertionError("Timed out waiting for smart-record callback"))
        except Exception as exc:
            failures.append(exc)
        finally:
            pipeline.stop()

    thread = threading.Thread(target=record_and_stop, daemon=True)
    thread.start()
    try:
        pipeline.start().wait()
    finally:
        pipeline.stop()
        thread.join()

    if failures:
        raise failures[0]
    assert result["container_type"] == "MP4"
    assert result["duration"] > 0
    assert os.path.exists(result["file_path"])
    assert result["contains_video"] is expected_video
    assert result["contains_audio"] is expected_audio
    if expected_video:
        assert result["width"] > 0
        assert result["height"] > 0
    if expected_audio:
        assert result["channels"] > 0
        assert result["sampling_rate"] > 0

def test_audio_encode_wav():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    Flow(Pipeline("test-audio-encode")).capture([AUDIO_FILE], media_types=MediaType.AUDIO).encode(OUTPUT_FILE)()

    mediainfo = MediaInfo.discover(OUTPUT_FILE)
    audio_streams = [stream for stream in mediainfo.streams if isinstance(stream, AudioStreamInfo)]
    assert len(audio_streams) == 1
    assert audio_streams[0].channels > 0
    assert audio_streams[0].sampling_rate > 0

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

def test_audio_batch_runs():
    pipeline = Pipeline("test-audio-batch")
    validator = AudioBatchValidator()
    thread = threading.Thread(target=stop_pipeline_on_event, args=(pipeline, validator.done), daemon=True)
    thread.start()

    Flow(pipeline).capture(
        [AV_SOURCE_FILE, AV_SOURCE_FILE],
        media_types=MediaType.AUDIO
    ).batch(
        batch_size=2
    ).attach(
        what=Probe("audio-batch-validator", validator)
    ).render(
        RenderMode.DISCARD,
        sync=False
    )()

    thread.join()
    if validator.failure:
        raise validator.failure
    assert not getattr(validator.done, "timed_out", False)
    assert validator.audio_batches > 0
    assert validator.audio_frames > 0

def test_audio_batch_capture_runs():
    pipeline = Pipeline("test-audio-batch-capture")
    validator = AudioBatchValidator()
    thread = threading.Thread(target=stop_pipeline_on_event, args=(pipeline, validator.done), daemon=True)
    thread.start()

    Flow(pipeline).batch_capture(
        [AUDIO_FILE, AUDIO_FILE],
        media_types=MediaType.AUDIO
    ).attach(
        what=Probe("audio-batch-capture-validator", validator)
    ).render(
        RenderMode.DISCARD,
        sync=False
    )()

    thread.join()
    if validator.failure:
        raise validator.failure
    assert not getattr(validator.done, "timed_out", False)
    assert validator.audio_batches > 0
    assert validator.audio_frames > 0

def test_audio_retrieve_runs():
    pipeline = Pipeline("test-audio-retrieve")
    retriever = AudioBufferRetriever()
    thread = threading.Thread(target=stop_pipeline_on_event, args=(pipeline, retriever.done), daemon=True)
    thread.start()

    Flow(pipeline).capture(
        [AUDIO_FILE],
        media_types=MediaType.AUDIO
    ).retrieve(
        retriever,
        audio_caps="audio/x-raw,format=S16LE,layout=interleaved"
    )()

    thread.join()
    assert not getattr(retriever.done, "timed_out", False)
    assert retriever.buffers > 0

def test_audio_inject_retrieve_runs():
    provider = AudioBufferProvider()
    retriever = AudioBufferRetriever()
    Flow(Pipeline("test-audio-inject-retrieve")).inject(
        [provider]
    ).retrieve(
        retriever,
        audio_caps="audio/x-raw,format=S16LE,rate=16000,channels=1,layout=interleaved"
    )()
    assert provider.buffers == provider.expected_buffers
    assert retriever.buffers > 0

def test_audio_infer_metadata():
    onnx_file = _infer_config_asset_path(SONYC_INFER_CONFIG, "onnx-file")
    if onnx_file and not onnx_file.exists():
        pytest.skip(f"SONYC audio model is not installed: {onnx_file}")

    pipeline = Pipeline("test-audio-infer")
    thread = threading.Thread(target=stop_pipeline, args=(pipeline,), daemon=True)
    thread.start()

    validator = AudioMetadataValidator()
    Flow(pipeline).capture(
        [AUDIO_FILE],
        media_types=MediaType.AUDIO
    ).batch(
        batch_size=1
    ).infer(
        SONYC_INFER_CONFIG,
        batch_size=1,
        audio_framesize=SONYC_AUDIO_FRAME_SIZE,
        audio_hopsize=SONYC_AUDIO_HOP_SIZE,
        audio_transform=SONYC_AUDIO_TRANSFORM,
        audio_caps=f"audio/x-raw, rate={SONYC_AUDIO_RATE}"
    ).attach(
        what=Probe("audio-metadata-validator", validator)
    ).render(
        RenderMode.DISCARD,
        sync=False
    )()

    thread.join()
    if validator.failure:
        raise validator.failure
    assert validator.audio_batches > 0
    assert validator.extracted_batches > 0
    assert validator.audio_frames > 0
    assert validator.acquire_frame_meta_rejected
    assert validator.event_messages_generated > 0

def test_audio_publish_msgconv_defaults():
    default_flow = Flow(Pipeline("test-audio-publish-default")).capture(
        [AUDIO_FILE],
        media_types=MediaType.AUDIO
    ).publish(
        msg_broker_proto_lib=MSG_BROKER_PROTO_LIB,
        msg_broker_conn_str="localhost;9092",
        topic="audio-test"
    )
    default_sink = _find_cached_node_name(default_flow.pipeline, "publish-msgsink-")
    assert default_flow.pipeline[default_sink].get("msg-conv-msg2p-lib") == DEFAULT_AUDIO_MSG2P_LIB

    custom_lib = "/tmp/libcustom_audio_msgconv.so"
    override_flow = Flow(Pipeline("test-audio-publish-override")).capture(
        [AUDIO_FILE],
        media_types=MediaType.AUDIO
    ).publish(
        msg_broker_proto_lib=MSG_BROKER_PROTO_LIB,
        msg_broker_conn_str="localhost;9092",
        topic="audio-test",
        msg_conv_msg2p_lib=custom_lib
    )
    override_sink = _find_cached_node_name(override_flow.pipeline, "publish-msgsink-")
    assert override_flow.pipeline[override_sink].get("msg-conv-msg2p-lib") == custom_lib

def test_audio_video_encode_mp4():
    if os.path.exists(AV_OUTPUT_FILE):
        os.remove(AV_OUTPUT_FILE)

    Flow(Pipeline("test-audio-video-encode")).capture(
        [AV_SOURCE_FILE],
        media_types=(MediaType.VIDEO, MediaType.AUDIO)
    ).encode(AV_OUTPUT_FILE)()

    mediainfo = MediaInfo.discover(AV_OUTPUT_FILE)
    audio_streams = [stream for stream in mediainfo.streams if isinstance(stream, AudioStreamInfo)]
    video_streams = [stream for stream in mediainfo.streams if isinstance(stream, VideoStreamInfo)]
    assert len(audio_streams) == 1
    assert audio_streams[0].channels > 0
    assert audio_streams[0].sampling_rate > 0
    assert len(video_streams) == 1
    assert video_streams[0].width > 0
    assert video_streams[0].height > 0

    if os.path.exists(AV_OUTPUT_FILE):
        os.remove(AV_OUTPUT_FILE)

def test_audio_video_select_audio_encode_wav():
    selected_output = "/tmp/pysm-audio-selected-test.wav"
    if os.path.exists(selected_output):
        os.remove(selected_output)

    Flow(Pipeline("test-audio-video-select-audio")).capture(
        [AV_SOURCE_FILE],
        media_types=(MediaType.VIDEO, MediaType.AUDIO)
    ).select(
        MediaType.AUDIO
    ).encode(
        selected_output
    )()

    mediainfo = MediaInfo.discover(selected_output)
    audio_streams = [stream for stream in mediainfo.streams if isinstance(stream, AudioStreamInfo)]
    assert len(audio_streams) == 1
    assert audio_streams[0].channels > 0
    assert audio_streams[0].sampling_rate > 0

    if os.path.exists(selected_output):
        os.remove(selected_output)

def test_audio_video_select_video_runs():
    pipeline = Pipeline("test-audio-video-select-video")
    thread = threading.Thread(target=stop_pipeline, args=(pipeline,), daemon=True)
    thread.start()

    Flow(pipeline).capture(
        [AV_SOURCE_FILE],
        media_types=(MediaType.VIDEO, MediaType.AUDIO)
    ).select(
        MediaType.VIDEO
    ).render(
        RenderMode.DISCARD,
        sync=False
    )()

    thread.join()

def test_audio_rejects_invalid_contracts():
    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-invalid-media-type")).capture([AUDIO_FILE], media_types="subtitle")

    with pytest.raises(Exception):
        Flow(Pipeline("test-audio-mixed-batch")).capture(
            [AV_SOURCE_FILE],
            media_types=(MediaType.VIDEO, MediaType.AUDIO)
        ).batch()

    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-mixed-batch-capture")).batch_capture(
            [AUDIO_FILE],
            media_types=(MediaType.VIDEO, MediaType.AUDIO)
        )

    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-triton-rejected")).capture(
            [AUDIO_FILE],
            media_types=MediaType.AUDIO
        ).infer(
            SONYC_INFER_CONFIG,
            with_triton=True
        )

    with pytest.raises(Exception):
        Flow(Pipeline("test-audio-analyze-rejected")).capture(
            [AUDIO_FILE],
            media_types=MediaType.AUDIO
        ).analyze("unused")

    with pytest.raises(Exception):
        Flow(Pipeline("test-audio-track-rejected")).capture(
            [AUDIO_FILE],
            media_types=MediaType.AUDIO
        ).track()

    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-rtsp-encode-rejected")).capture(
            [AUDIO_FILE],
            media_types=MediaType.AUDIO
        ).encode("rtsp://localhost:8554/audio")

    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-extension-rejected")).capture(
            [AUDIO_FILE],
            media_types=MediaType.AUDIO
        ).encode("/tmp/pysm-audio-test.ogg")

def test_audio_provider_validation():
    provider = AudioBufferProvider()
    provider.media_type = "video"
    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-provider-media-type")).inject([provider])

    provider = AudioBufferProvider()
    provider.format = "RGB"
    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-provider-format")).inject([provider])

    provider = AudioBufferProvider()
    del provider.sample_rate
    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-provider-rate")).inject([provider])

    provider = AudioBufferProvider()
    del provider.channels
    with pytest.raises(TypeError):
        Flow(Pipeline("test-audio-provider-channels")).inject([provider])

if __name__ == '__main__':
    test_audio_encode_wav()
