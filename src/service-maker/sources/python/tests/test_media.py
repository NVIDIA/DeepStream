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

from pyservicemaker import BufferProvider, ColorFormat, Buffer, Flow, Pipeline, RenderMode
from pyservicemaker.utils import MediaExtractor, MediaChunk, MediaInfo, AudioStreamInfo, VideoStreamInfo, StreamFrameRetriever
import queue
import concurrent.futures
import os
import threading

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD

JPEG_FILE = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg"
VIDEO_FILE = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"
AUDIO_FILE = "/opt/nvidia/deepstream/deepstream/samples/streams/sonyc_mixed_audio.wav"
N_CHUNKS = 8
N_THREADS = 2

class MyBufferProvider(BufferProvider):
    def __init__(self, q, width, height):
        super().__init__()
        self._queue = q
        self.format = "RGB"
        self.width = width
        self.height = height
        self.framerate = 30
        self.device = 'gpu'
        self.frames = 0

    def generate(self, size):
        try:
            frame = self._queue.get(timeout=20)
            if frame is None:
                print(f"End of Queue, {self.frames} collected")
                return Buffer()
            self.frames += 1
            return frame.tensor.wrap(ColorFormat.RGB)
        except queue.Empty:
            print(f"Buffer queue empty, {self.frames} collected")
            return Buffer()

def test_context():
    n_frames = 0
    FRAMES = 20
    with MediaExtractor(chunks=[MediaChunk(VIDEO_FILE)]) as m:
        qs = m()
        assert(len(qs) == 1)
        for i in range(FRAMES):
            try:
                if (qs[0].get(timeout=2) is None):
                    print("End of Queue")
                    break
                n_frames += 1
            except queue.Empty:
                print("Buffer queue empty")
                break
    assert(n_frames == FRAMES)

def test_extract_single_jpeg():
    qs = MediaExtractor(chunks=[MediaChunk(JPEG_FILE)])()
    assert(len(qs) == 1)
    p = MyBufferProvider(qs[0], 1280, 720)
    Flow(Pipeline("test")).inject([p]).render(RENDER_MODE, sync=False)()

def test_extract_single_video():
    qs = MediaExtractor(chunks=[MediaChunk(VIDEO_FILE)])()
    assert(len(qs) == 1)
    Flow(Pipeline("renderer")).inject([MyBufferProvider(qs[0], 1280, 720)]).render(RENDER_MODE, sync=False)()

def test_duration():
    qs = MediaExtractor(chunks=[MediaChunk(VIDEO_FILE, 4*1e9, 1e9)])()
    assert(len(qs) == 1)
    Flow(Pipeline("renderer")).inject([MyBufferProvider(qs[0], 1280, 720)]).render(RENDER_MODE, sync=False)()

def test_extract_single_video_seek():
    qs = MediaExtractor(chunks=[MediaChunk(VIDEO_FILE, start_pts=100000000000)])()
    assert(len(qs) == 1)
    Flow(Pipeline("renderer")).inject([MyBufferProvider(qs[0], 1280, 720)]).render(RENDER_MODE, sync=False)()

def test_batch_extract():
    chunks = [MediaChunk(f) for f in [VIDEO_FILE]*N_CHUNKS]
    for batch_size in [0, 2, 4, 8]:
        qs = MediaExtractor(chunks=chunks, batch_size=batch_size, n_thread=N_CHUNKS/batch_size if batch_size else N_CHUNKS)()
        assert(len(qs) == N_CHUNKS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_CHUNKS) as exe:
            exe.map(lambda q: Flow(Pipeline("renderer")).inject([MyBufferProvider(q, 1280, 720)]).render(RENDER_MODE, sync=False)(), qs)

def test_mediainfo():
    mediainfo = MediaInfo.discover(JPEG_FILE)
    assert(len(mediainfo.streams) == 1)
    assert(isinstance(mediainfo.streams[0], VideoStreamInfo))
    stream = mediainfo.streams[0]
    assert(stream.codec == 'JPEG')
    assert(stream.width > 0)
    assert(stream.height > 0)
    mediainfo = MediaInfo.discover(VIDEO_FILE)
    assert(mediainfo.duration != 0)
    v_streams = [s for s in mediainfo.streams if isinstance(s, VideoStreamInfo)]
    assert(len(v_streams) == 1)
    assert('264' in v_streams[0].codec)
    assert(v_streams[0].width > 0)
    assert(v_streams[0].height > 0)
    mediainfo = MediaInfo.discover(AUDIO_FILE)
    a_streams = [s for s in mediainfo.streams if isinstance(s, AudioStreamInfo)]
    assert(len(a_streams) == 1)
    assert(a_streams[0].channels > 0)
    assert(a_streams[0].sampling_rate > 0)

def test_frame_resampling():
    chunks = [MediaChunk(source=VIDEO_FILE, interval=2*1e9)]
    with MediaExtractor(chunks=chunks) as m:
        qs = m()
    assert(len(qs) == 1)
    Flow(Pipeline("renderer")).inject([MyBufferProvider(qs[0], 1280, 720)]).render(RENDER_MODE, sync=False)()

def test_dynamic_source():
    def append_concurrently(extractor, chunks):
        start = threading.Barrier(len(chunks))

        def append_chunk(chunk):
            start.wait()
            return extractor.append(chunk)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as exe:
            return list(exe.map(append_chunk, chunks))

    with MediaExtractor(n_thread=N_THREADS) as m:
        m()

        chunks = [MediaChunk(source=VIDEO_FILE) for _ in range(N_CHUNKS)]
        qs = append_concurrently(m, chunks)
        assert([len(r._chunks) for r in m._retrievers] == [N_CHUNKS // N_THREADS] * N_THREADS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_CHUNKS) as exe:
            exe.map(lambda q: Flow(Pipeline("renderer")).inject([MyBufferProvider(q, 1280, 720)]).render(RENDER_MODE, sync=False)(), qs)

        chunks = [MediaChunk(source=VIDEO_FILE, duration=10*1e9, interval=1e9) for _ in range(N_CHUNKS)]
        qs = append_concurrently(m, chunks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_CHUNKS) as exe:
            exe.map(lambda q: Flow(Pipeline("renderer")).inject([MyBufferProvider(q, 1280, 720)]).render(RENDER_MODE, sync=False)(), qs)

def test_stream_retriever_handles_source_completion():
    retriever = StreamFrameRetriever(queue.Queue(), 0)
    first = MediaChunk(VIDEO_FILE)
    first.id = 1
    second = MediaChunk(VIDEO_FILE)
    second.id = 2
    first_q = queue.Queue(maxsize=1)
    second_q = queue.Queue(maxsize=1)

    retriever.add_chunk(first, first_q)
    retriever.add_chunk(second, second_q)
    retriever._sampler = object()

    assert retriever.finish_chunk(1) is True
    assert retriever._sampler is None
    assert first_q.get(timeout=1) is None
    assert [chunk.id for chunk in retriever._chunks] == [2]

    assert retriever.finish_chunk(2) is True
    assert second_q.get(timeout=1) is None
    assert retriever._chunks == []

    assert retriever.finish_chunk(99) is False

if __name__ == '__main__':
    test_dynamic_source()
    test_mediainfo()
    test_context()
    test_duration()
    test_extract_single_jpeg()
    test_extract_single_video()
    test_extract_single_video_seek()
    test_batch_extract()
    test_frame_resampling()
