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

from dataclasses import dataclass
from .pipeline import Pipeline
from .flow import Flow
from .logging import get_logger
from ._pydeepstream.utils import *
from ._pydeepstream import BufferRetriever, DynamicSourceMessage, Tensor
from ._pydeepstream.signal import SourceManager

from typing import List, Tuple, Callable, Optional, Type
from types import TracebackType
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

from enum import Enum

logger = get_logger(__name__)

class MediaChunk:
    """Carries information of a chunk in a media source"""
    def __init__(self, source: str, start_pts: int=0, duration: int=-1, interval: int=0):
        """
        Args:
            source: file path or url of a media source
            start_pts: timestamp in nano-second for start
            duration:  duration in nano-second
            interval: frame sampling interval in nano-second, 0 for no frame skipping
        """
        self._source = source
        self._start_pts = start_pts
        self._duration = duration
        self._interval = interval

    @property
    def source(self):
        return self._source

    @property
    def start_pts(self):
        return self._start_pts

    @property
    def duration(self):
        return self._duration

    @property
    def interval(self):
        return self._interval

class VideoFrame:
    """Represents a decoded video frame"""
    def __init__(self, data: Tensor, timestamp: int=-1):
        self._timestamp: int = timestamp
        self._tensor: Tensor = data

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def tensor(self):
        return self._tensor

class FrameSampler:
    """Manages frame sampling"""
    def __init__(self, chunk: MediaChunk, seek_fn: Callable = None):
        self._seek_enabled = True if seek_fn else False
        self._seek_fn = seek_fn
        self._last_good_seek = 0
        self._last_seek_pts = 0
        self._last_pts = 0
        self._n_frames = 0
        self._chunk = chunk
        self._done = False

    @property
    def done(self):
        return self._done

    def sample(self, buffer, pts):
        if self._chunk._start_pts == 0:
            self._chunk._start_pts = pts
        if self._chunk.duration > 0 and pts > self._chunk._start_pts + self._chunk.duration:
            logger.info(f"Media chunk for {self._chunk.source} is done")
            self._done = True
            return None

        expected_pts = self._chunk._start_pts + self._chunk.interval * (self._n_frames) if self._chunk.interval > 0 else pts
        frame = None
        if pts < self._chunk._start_pts:
            logger.info(f"Frame skipped: pts: {pts} < start_pts: {self._chunk._start_pts}")
        elif pts >= expected_pts:
            frame = VideoFrame(buffer.extract(0).clone(), pts)
            self._n_frames += 1
            logger.info(f"Frame captured: {self._n_frames}, pts: {pts}")
        else:
            logger.info(f"Frame skipped: expected pts: {expected_pts}, actual pts: {pts}")
            if self._seek_enabled:
                # evaluate the seeking efficiency
                if self._last_pts > pts:
                    logger.warning("Seeking disabled due to less efficient frame retrieval")
                    self._seek_enabled = False
                elif pts - self._last_pts > 2 * self._chunk.interval:
                    logger.warning("Seeking disabled due to large frame gap")
                    self._seek_enabled = False
                    self._seek_fn(self._last_good_seek)
            if self._seek_enabled and expected_pts - pts > 1e9 and self._last_seek_pts != expected_pts:
                logger.info(f"Seeking to {expected_pts}")
                self._seek_fn(expected_pts)
                self._last_seek_pts = expected_pts
        self._last_pts = pts
        return frame

class StreamFrameRetriever(BufferRetriever):
    SEEK_THRESHOLD = 1e9 # 1 second
    SEEK_GAP = 2 # 2 intervals

    def __init__(
            self,
            event_queue: queue.Queue,
            id,
            seek_enabled: bool=False
    ):
        super().__init__()
        self._event_queue = event_queue
        self._pipeline_id = id
        self._seek_enabled = seek_enabled
        self._chunks = []
        self._queues = []
        self._sampler = None
        self._lock = threading.Lock()

    @staticmethod
    def _signal_queue_done(q: queue.Queue):
        try:
            q.put_nowait(None)
        except queue.Full:
            threading.Thread(target=q.put, args=(None,), daemon=True).start()

    def add_chunk(self, chunk: MediaChunk, q: queue.Queue):
        with self._lock:
            self._chunks.append(chunk)
            self._queues.append(q)

    def finish_chunk(self, source_id: int):
        with self._lock:
            for idx, chunk in enumerate(list(self._chunks)):
                if getattr(chunk, "id", None) != source_id:
                    continue

                q = self._queues[idx]
                del self._chunks[idx]
                del self._queues[idx]
                if idx == 0:
                    self._sampler = None
                self._signal_queue_done(q)
                logger.info(f"Chunk {source_id} completed on source EOS")
                return True

        logger.info(f"Ignoring EOS for unknown or already completed chunk {source_id}")
        return False

    def consume(self, buffer):
        with self._lock:
            if not self._chunks:
                logger.warning("No media chunk in the retriever")
                return 0
            chunk_id = buffer.get_chunk_id(0)
            # work on the current chunk
            chunk = self._chunks[0]
            pending_chunk = None if len(self._chunks) == 1 else self._chunks[1]
            pending_q = None if len(self._queues) == 1 else self._queues[1]
            q = self._queues[0]
            if chunk_id != chunk.id:
                if pending_chunk is not None and pending_chunk.id == chunk_id:
                    logger.info(f"New chunks arriving, chunk id: {chunk_id}")
                    # reset the retriever
                    self._chunks.remove(chunk)
                    self._signal_queue_done(q)
                    self._queues.remove(q)
                    self._sampler = None
                    chunk = pending_chunk
                    q = pending_q
                elif not self._sampler:
                    logger.info(f"Dropping stashed frames from old chunk {chunk_id}")
                    return 1
                else:
                    logger.warning(f"Unexpected chunk id: {chunk_id} with timestamp {buffer.timestamp}, current: {chunk.id}")
                    return 1
            if not self._sampler:
                logger.info(f"Creating a new sampler for new chunk {chunk.id}")
                self._sampler = FrameSampler(
                    chunk,
                    None if not self._seek_enabled else lambda pts: self._event_queue.put(
                        MediaExtractor.Event(MediaExtractor.EventType.SEEK, self._pipeline_id, pts)
                    )
                )
            frame = self._sampler.sample(buffer, buffer.timestamp)
            if frame:
                try:
                    q.put(frame, timeout=1.0)
                except queue.Full:
                    logger.warning("Queue is full. Dropping frame.")
            elif self._sampler.done:
                logger.info(f"Chunk {chunk.id} completed")
                self._event_queue.put(
                    MediaExtractor.Event(MediaExtractor.EventType.REMOVE_SOURCE, self._pipeline_id, self._chunks[0].id)
                )
                # our goal on current chunk achieved, reset the retriever
                self._chunks.remove(chunk)
                self._signal_queue_done(q)
                self._queues.remove(q)
                self._sampler = None
        return 1

    def stop(self):
        with self._lock:
            queues = self._queues
            self._chunks = []
            self._queues = []
            self._sampler = None

        for q in queues:
            self._signal_queue_done(q)

class BulkFrameRetriever(BufferRetriever):

    SEEK_THRESHOLD = 1e9 # 1 second
    SEEK_GAP = 2 # 2 intervals

    def __init__(
            self,
            event_queue: queue.Queue,
            id,
            chunks: List[MediaChunk],
            qs: list[queue.Queue],
            seek_enabled: bool=False,
            blocking_mode: bool=False
    ):
        super().__init__()
        self._event_queue = event_queue
        self._pipeline_id = id
        self._chunks = chunks
        self._queues = qs
        self._dones = [False] * len(chunks)
        self._stopped = False
        seek_fn = None if not seek_enabled else lambda pts: self._event_queue.put(
            MediaExtractor.Event(MediaExtractor.EventType.SEEK, self._pipeline_id, pts)
        )
        self._samplers = [FrameSampler(chunk=c, seek_fn=seek_fn) for c in chunks]
        self._blocking_mode = blocking_mode


    def consume(self, buffer):
        if self._stopped :
            return 0
        if (len(self._queues) == 1):
            sampler = self._samplers[0]
            frame = sampler.sample(buffer, buffer.timestamp)
            if frame:
                while not self._stopped:
                    try:
                        self._queues[0].put(frame, timeout=1.0)
                        break
                    except queue.Full:
                        if not self._blocking_mode:
                            logger.warning("Queue is full. Dropping frame.")
                            break
                        else:
                            logger.info("Queue is full. Waiting for frame to be consumed.")
            if sampler.done:
                self._dones[0] = sampler.done
                if not self._queues[0].full():
                    self._queues[0].put(None)
        else:
            for frame_meta in buffer.batch_meta.frame_items:
                id = frame_meta.pad_index
                sampler = self._samplers[id]
                frame = sampler.sample(buffer, frame_meta.buffer_pts)
                if frame:
                    while not self._stopped:
                        try:
                            self._queues[id].put(frame, timeout=1.0)
                            break
                        except queue.Full:
                            if not self._blocking_mode:
                                logger.warning("Queue is full. Dropping frame.")
                                break
                            else:
                                logger.info("Queue is full. Waiting for frame to be consumed.")
                if sampler.done:
                    self._dones[id] = sampler.done
                    if not self._queues[id].full():
                        self._queues[id].put(None)
        if all(self._dones) and not self._stopped:
            for i in range(len(self._queues)):
                self._event_queue.put(MediaExtractor.Event(MediaExtractor.EventType.STOP, i, None))
            self._stopped = True
        return 1

    def stop(self):
        if self._stopped:
            # already stopped
            return
        self._stopped = True
        for q in [q for q in self._queues if not q.full()]:
            # signal the end of the queue
            q.put(None)

class MediaExtractor:
    """Callable for extract data from media chunks"""
    DEFAULT_RESOLUTION = (1920, 1080)

    class EventType(Enum):
        STOP = "stop"
        SEEK = "seek"
        REMOVE_SOURCE = "remove_source"

    @dataclass
    class Event:
        type: 'MediaExtractor.EventType'
        id: int
        data: any


    """
        Args:
            chunks    : a list of video chunk specification as input
            batch_size: 0 means no batch. once set to N, decoded frames will be batched and scaled
            scaling   : target resolution of (width, height), not applicable if batch_size is 0
            n_thread  : number of worker threads
            q_size    : the capacity of the output queue,
            enable_seek: enable seeking for frame retrieval
            blocking: if True, the frame retriever will block when the decoded queue is full
    """
    def __init__(
            self,
            chunks: List[MediaChunk]|None=None,
            batch_size: int=0,
            scaling: Tuple=DEFAULT_RESOLUTION,
            n_thread: int=1,
            q_size: int=1,
            enable_seek: bool=False,
            blocking: bool=False
        ):
        self._pipelines = []
        self._flows = []
        self._retrievers = []
        self._chunks = chunks if chunks else []
        self._queues = [queue.Queue(maxsize=q_size) for _ in self._chunks]
        self._source_managers = []
        self._auto_index = 0 # index to pick the idle thread.
        self._max_q_size = q_size
        self._append_lock = threading.Lock()

        # create a event thread to handle events
        def handle_events(self):
            while self._pipelines:
                try:
                    event = self._event_queue.get(timeout=1)
                    if event.id < 0 or event.id > (len(self._pipelines) - 1):
                        logger.warning(f"Pipeline id not found on event {event}")
                        continue
                    if event.type == MediaExtractor.EventType.STOP:
                        logger.info(f"Handling Stop Event {event}")
                        pipeline = self._pipelines[event.id]
                        pipeline.stop()
                        continue
                    if event.type == MediaExtractor.EventType.SEEK:
                        logger.info(f"Handling Seek Event: {event}")
                        pipeline = self._pipelines[event.id]
                        pipeline.seek(float(event.data/1e9))
                        continue
                    if event.type == MediaExtractor.EventType.REMOVE_SOURCE:
                        logger.info(f"Handling Remove Source Event: {event}")
                        self._source_managers[event.id].remove_source(event.data)
                        continue
                except queue.Empty:
                    continue
        self._event_queue = queue.Queue()

        if chunks:
            # associate the chunks with the queues
            whole_list_of_chunks = [[c] for c in chunks]
            whole_list_of_queues = [[q] for q in self._queues]
            if batch_size > 0:
                # split the chunk list by batch size
                whole_list_of_chunks = [chunks[i: i + batch_size] for i in range(0, len(chunks), batch_size)]
                whole_list_of_queues = [self._queues[i: i + batch_size] for i in range(0, len(self._queues), batch_size)]
            for chunk_list, q_list in zip(whole_list_of_chunks, whole_list_of_queues):
                pipeline = Pipeline(f"media_extractor_pipeline")
                pipeline_id = len(self._pipelines)
                r = BulkFrameRetriever(self._event_queue, pipeline_id, chunk_list, q_list, enable_seek, blocking)
                if batch_size == 0:
                    assert(len(chunk_list) == 1)
                    flow = Flow(pipeline).capture([chunk_list[0].source]).retrieve(r)
                    if (chunk_list[0].start_pts > 0):
                        pipeline.seek(float(chunk_list[0].start_pts/1e9))
                else:
                    sources = [chunk.source for chunk in chunk_list]
                    flow = Flow(pipeline).batch_capture(sources, width=scaling[0], height=scaling[1]).retrieve(r)
                self._flows.append(flow)
                self._pipelines.append(pipeline)
                self._retrievers.append(r)
        else:
            # The application is responsible for appending sources during runtime
            for i in range(n_thread):
                pipeline = Pipeline(f"media_extractor_pipeline_{i}")
                pipeline_id = len(self._pipelines)
                r = StreamFrameRetriever(self._event_queue, pipeline_id, enable_seek)
                source_manager = SourceManager(f"source_manager_{i}")
                flow = Flow(pipeline).capture([], source_manager).retrieve(r)
                self._flows.append(flow)
                self._pipelines.append(pipeline)
                self._retrievers.append(r)
                self._source_managers.append(source_manager)

        # start the event thread
        self._event_thread = threading.Thread(target=handle_events, daemon=True, args=(self,))
        self._event_thread.start()

        # create a thread pool and futures
        self._futures = []
        self._executor = ThreadPoolExecutor(max_workers=n_thread)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
    ) -> None:
        for s in self._source_managers:
            s.terminate()
        self._source_managers.clear()
        for r in self._retrievers:
            r.stop()
        self._retrievers.clear()
        for p in self._pipelines:
            p.stop()
        self._pipelines.clear()
        self._executor.shutdown()

    def __call__(self):
        def _run_flow(flow, retriever):
            def _on_message(message):
                if (
                    isinstance(message, DynamicSourceMessage) and
                    not message.source_added and
                    hasattr(retriever, "finish_chunk")
                ):
                    retriever.finish_chunk(message.source_id)

            if hasattr(retriever, "finish_chunk"):
                flow(_on_message)
            else:
                flow()
            # signal the end of the queue
            retriever.stop()
            # remove the retriever from the list
            if retriever in self._retrievers:
                self._retrievers.remove(retriever)
            # remove the pipeline from the list
            if flow.pipeline in self._pipelines:
                self._pipelines.remove(flow.pipeline)

        for flow, retriever in zip(self._flows, self._retrievers):
            future = self._executor.submit(_run_flow, flow, retriever)
            self._futures.append(future)
        return self._queues

    def __del__(self):
        for s in self._source_managers:
            s.terminate()
        self._source_managers.clear()
        for r in self._retrievers:
            r.stop()
        self._retrievers.clear()
        for p in self._pipelines:
            p.stop()
        self._pipelines.clear()
        self._executor.shutdown()

    def append(self, chunk: MediaChunk):
        if self._source_managers:
            with self._append_lock:
                pipeline_index = self._auto_index
                q = queue.Queue(self._max_q_size)
                self._retrievers[pipeline_index].add_chunk(chunk, q)
                chunk.id = self._source_managers[pipeline_index].add_source(chunk.source)
                self._auto_index = (pipeline_index + 1) % len(self._source_managers)
                return q
        else:
            raise RuntimeError("Dynamic adding source is not supported with this instance")
