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


import os
import re
import sys
import logging
import time
from typing import Callable, Optional, List, Dict, Any
import importlib
import numpy as np
import torch
import jinja2
import json
from queue import Queue, Full, Empty
from concurrent.futures import Future
import threading

kDebug = int(os.getenv("DEBUG", "0"))
PACKAGE_NAME = "ib"

class NumpyFlatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        return super().default(obj)


def concat_tensors_in_dict(list_of_tensor_dicts: List) -> Dict:
    result = {}

    # Iterate over each dictionary in the list
    for d in list_of_tensor_dicts:
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                result[key] = np.concatenate(result[key], value) if key in result else value
            elif isinstance(value, torch.Tensor):
                result[key] = torch.cat((result[key], value)) if key in result else value
            else:
                result[key] = result[key] + value if key in result else value
    return result

def stack_tensors_in_dict(list_of_tensor_dicts: List) -> Dict:
    """
    [{'k1': 'v1', 'k2': 'v2'}, {'k1': 'v3', 'k2': 'v4'}] ->
    {'k1': ['v1', 'v3'], 'k2': ['v2', 'v4']}
    """
    result = {}

    # Iterate over each dictionary in the list
    for d in list_of_tensor_dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = []  # Create a new list for each new key
            result[key].append(value)  # Append the value to the list
    # Iterate over the combined dictionary
    for key in result:
        tensor_list = result[key]
        if isinstance(tensor_list[0], np.ndarray):
            result[key] = np.stack(tensor_list, axis=0)
        elif isinstance(tensor_list[0], torch.Tensor):
            result[key] = torch.stack(tensor_list, dim=0)

    return result

def split_tensor_in_dict(dict_of_tensor_list: Dict) -> List:
    """
    {'k1': ['v1', 'v2'], 'k2': ['v3', 'v4']} ->
    [{'k1': 'v1', 'k2': 'v3'}, {'k1': 'v2', 'k2': 'v4'}]
    """
    values = [dict_of_tensor_list[k] for k in dict_of_tensor_list]
    result = []

    def _len(v):
        if isinstance(v, list):
            return len(v)
        if isinstance(v, np.ndarray) and v.ndim > 0:
            return len(v)
        return 1

    length = min(_len(v) for v in values)
    for i in range(length):
        result.append({
            k: v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v[i]
            for k, v in dict_of_tensor_list.items()
        })

    return result


def convert_list(i: List, f: Callable):
    # If the item is a list, apply the function recursively
    if isinstance(i, list):
        return [convert_list(item, f) for item in i]
    else:
        return f(i)

def import_class(module_name, class_name):
    # Import the module using importlib
    module = importlib.import_module(module_name)

    # Get the class from the module using getattr
    class_ = getattr(module, class_name)

    return class_

def create_jinja2_env():
    def start_with(field, s):
        return field.startswith(s)

    def replace(value, pattern, text):
        return re.sub(pattern, text, value)

    def extract(value, pattern):
        match = re.search(pattern, value)
        return match.group(1) if match else ''

    def raise_helper(message):
        raise Exception(message)

    def tolist(value):
        if hasattr(value, 'tolist'):
            # Convert NumPy array to list and flatten to 1D
            return value.flatten().tolist()
        elif isinstance(value, list):
            # Already a list, return as is
            return value
        else:
            # Fallback: try to convert to list
            return list(value)

    jinja2_env = jinja2.Environment()
    jinja2_env.tests["startswith"] = start_with
    jinja2_env.filters["replace"] = replace
    jinja2_env.filters["extract"] = extract
    jinja2_env.filters["tolist"] = tolist
    jinja2_env.filters["zip"] = zip
    jinja2_env.globals["raise"] = raise_helper

    return jinja2_env

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """ Get component logger

        Parameters:
        name: a module name

        Returns: A Logger Instance
    """

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    if kDebug:
        log_level = logging.DEBUG
    name = f"{PACKAGE_NAME}.{name}"
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    # sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print(f"logger {str(name)}, log_level: {str(log_level)}")
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    logger = logging.getLogger(name)
    logger.propagate = True
    # formatter = logging.Formatter(log_format)
    # stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setFormatter(formatter)
    # stream_handler.setLevel(log_level)
    # logger.handlers.clear()
    # logger.addHandler(stream_handler)
    return logger

def flush(logger):
    for h in logger.handlers:
        h.flush()


class SimpleLogger:
    def __init__(self, name: str = ""):
        self.name = name
    def log_print(self, level, *args, **kwargs):
        asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        prefix = f"{asctime} [{str(level)}] [{self.name}]: "
        print(prefix, *args, **kwargs)

    def debug(self, *args, **kwargs):
        if kDebug:
            self.log_print("DEBUG", *args, **kwargs)

    def info(self, *args, **kwargs):
        self.log_print("INFO", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self.log_print("WARNING", *args, **kwargs)

    def error(self, *args, **kwargs):
        self.log_print("ERROR", *args, **kwargs)

def tensor_info(tensor):
    return f" shape: {tensor.shape}, dtype: {tensor.dtype}, dtype: {tensor.device}"


logger = get_logger(__name__)

class FutureWrapper:
    """A future wrapper for collecting results from an asynchronous operation"""
    def __init__(self, future: Future, depot: Queue, *user_data):
        self._future = future
        self._depot = depot
        self._user_data = list(user_data)
        self._future.add_done_callback(self._on_done)


    def __repr__(self):
        return f"FutureWrapper(done={self._future.done()})"

    def _on_done(self, future: Future):
        try:
            self._depot.put_nowait(self)
        except Full:
            # fallback policy: best-effort enqueue with small timeout or log-and-drop
            try:
                self._depot.put(self, timeout=1)
            except Full:
                logger.warning(
                    f"FutureWrapper failed to enqueue to depot, dropped"
                    "consider increasing the future depot size"
                )

    @property
    def user_data(self):
        return self._user_data

    def result(self):
        return self._future.result()

    def done(self):
        return self._future.done()

    def cancel(self):
        self._future.cancel()

class FutureConsumer:
    def __init__(self,
                 name: str,
                 stop_event: threading.Event,
                 result_callback: Callable[[Any, Any], None]):
        self._name = name
        self._stop_event = stop_event
        self._future_depot = Queue()
        self._future_lock = threading.Lock()
        self._future_list = []
        self._result_callback = result_callback
        self._future_consumer_thread = threading.Thread(
                    target=self.run,
                    daemon=True
                )
        self._future_consumer_thread.start()

    def append_future(self, future: Future, *user_data):
        future_wrapper = FutureWrapper(future, self._future_depot, *user_data)
        with self._future_lock:
            self._future_list.append(future_wrapper)

    def run(self):
        if self._stop_event is None:
            logger.error(f"Future consumer {self._name} failed to start, stopped event is None")
            return
        if not callable(self._result_callback):
            logger.error(f"Future consumer {self._name} failed to start, result callback is not callable")
            return

        logger.info(f"Future consumer for {self._name} started")
        # Keep consuming completion signals and emit results in FIFO order of submission
        while not self._stop_event.is_set():
            try:
                # Wait for any future to complete; we only use this as a signal
                future_wrapper = self._future_depot.get(timeout=1)
            except Empty:
                continue
            except Exception as e:
                logger.exception(e)
                continue

            # Flush all completed futures from the head of the list (preserving order)
            while True:
                with self._future_lock:
                    if not self._future_list:
                        logger.info(f"Future consumer {self._name} future list is empty.")
                        break
                    if not self._future_list[0].done():
                        logger.info(f"Future consumer {self._name} the first future in the list is not actually done. {self._future_list}")
                        break
                    future_wrapper = self._future_list.pop(0)
                    self._result_callback(future_wrapper.result(), future_wrapper.user_data)

        logger.info(f"Future consumer for {self._name} stopped")

class ResultQueue:
    def __init__(self,
                 queue: Queue,
                 result_callback: Callable[[Any, Any], None],
                 user_data: Any,
                 stop_event: Optional[threading.Event] = None):
        self._queue = queue
        self._result_callback = result_callback
        self._user_data = user_data
        self._stop_event = stop_event

    def __repr__(self):
        return f"ResultQueue(queue={self._queue}, result_callback={self._result_callback}, user_data={self._user_data})"

    def get_all(self):
        while self._stop_event is None or not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                # nothing to drain right now
                continue
            except Exception as e:
                logger.exception(e)
                break
            self._result_callback(item, self._user_data)
            if not item:
                break
        logger.debug(f"ResultQueue drained")

class QueueConsumer:
    """Consume ResultQueues in a single daemon thread and emit results.

    Submit work via append_queue(queue, result_callback, user_data=None).
    Items pulled from the provided `queue` will be forwarded to
    `result_callback(item, user_data)`.
    """

    def __init__(self,
                 name: str,
                 stop_event: threading.Event,
                 queue_get_timeout: float = 1.0):
        self._name = name
        self._stop_event = stop_event
        self._queue_get_timeout = queue_get_timeout
        self._queue_depot = Queue()
        self._queue_consumer_thread = threading.Thread(target=self.run, daemon=True)
        self._queue_consumer_thread.start()

    def append_queue(self,
                     queue: Queue,
                     result_callback: Callable[[Any, Any], None],
                     user_data: Any = None):
        if not callable(result_callback):
            logger.error(f"Queue consumer {self._name} append_queue failed: result_callback is not callable")
            return
        result_queue = ResultQueue(queue, result_callback, user_data, self._stop_event)
        self._queue_depot.put(result_queue)

    def run(self):
        if self._stop_event is None:
            logger.error(f"Queue consumer {self._name} failed to start, stopped event is None")
            return

        logger.info(f"Queue consumer for {self._name} started")
        while not self._stop_event.is_set():
            try:
                q = self._queue_depot.get(timeout=self._queue_get_timeout)
            except Empty:
                continue
            except Exception as e:
                logger.exception(e)
                continue
            q.get_all()

        logger.info(f"Queue consumer for {self._name} stopped")
