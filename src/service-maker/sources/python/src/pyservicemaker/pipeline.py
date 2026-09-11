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

from typing import Dict
from ._pydeepstream import Pipeline as _Pipeline, Probe, Receiver, Feeder

class Pipeline:
    """
    The Pipeline class for media processing
    """
    def __init__(self, name: str, config_file: str = None):
        """
        Args:
            name (str):  name of the pipeline instance
            config_file: path to the yaml file for defining a pipeline in graphic
                         structure
        """
        self._instance = _Pipeline(name, config_file) if config_file else _Pipeline(name)
        # node cache for reference keeping
        self._nodes = dict()

    def __getitem__(self, name):
        """
        Get the item within the pipeline by name

        Args:
            name (str): name of the item given while being added

        Returns:
            item instance, set/get can be used for property access

        Raises:
            None
        """
        if name in self._nodes:
            return self._nodes[name]
        node = self._instance[name]
        self._nodes[name] = node
        return node

    def add(self, type_name: str, name: str, properties: Dict = None):
        """
        Create a processing node from Deepstream plugins and add it to the pipeline

        Args:
            type_name (str): type name registered through Deepstream
            name (str):      name of the instance, can be used for addressing it

        Returns:
            Pipeline instance

        Raises:
            None
        """
        self._instance.add(type_name, name)
        if properties:
            self[name].set(properties)
        return self

    def link(self, *args):
        """
        Link all the processing nodes within the pipeline by order

        Args:
            *args: names of the nodes to be linked in order, or two tuples indicating
                   source name and target name, source hint and target hint 

        Returns:
            Pipeline instance

        Raises:
            Exception: Invalid number of arguments
        """
        if len(args) < 2:
            raise Exception("Invalid number of arguments")

        if len(args) == 2 and isinstance(args[0], tuple) and isinstance(args[1], tuple):
            source = self[args[0][0]]
            sink = self[args[0][1]]
            source.link(sink, args[1])
        else:
            source = self[args[0]]
            for arg in args[1:]:
                sink = self[arg]
                source.link(sink)
                source = sink
        return self

    def attach(self, target: str, what, name="", tips="", properties: Dict=None):
        """
        Create and attach a custom object to the pipeline, the object can be
        created explicitly or from the specific plugin.
        The object can be a Probe, Receiver or Feeder.

        Args:
            target (str): name of the node to which the object is attached within the pipeline
            what :        an object or name of the plugin which creates the object
            name (str):   name of the object, not required for explicitly created object
            tips (str):   extra information for the custom object
            properties (Dict): properties to be set on the object, not applicable for explicitly created object

        Returns:
            Pipeline instance

        Raises:
            None
        """
        if (
          isinstance(what, Probe) or
          isinstance(what, Receiver) or
          isinstance(what, Feeder)
        ):
            self[target].attach(what, tips)
        elif (isinstance(what, str)):
            self._instance.attach(target, what, name, tips)
            if properties:
                object = self[target].find(name)
                object.set(properties)
        else:
            raise Exception("Unsupported object type for Pipeline.attach")
        return self


    def start(self, on_message=None):
        """
        Start the pipeline

        Args:
            on_message (callable): message callback, Callable[Pipeline, PipelineMessage]
        Returns:
            Pipeline instance

        Raises:
            None
        """
        if on_message is None:
            self._instance.start()
        elif callable(on_message):
            self._instance.start_with_callback(on_message)
        else:
            raise Exception("Message callback is not callable")
        return self
    
    def prepare(self, on_message=None):
        """
        Intialize the pipeline

        Args:
            on_message (callable): message callback, Callable[Pipeline, PipelineMessage]
        Returns:
            int value
            1 : success
            -1: failure

        Raises:
            None
        """
        if on_message is None:
            self._instance.prepare()
        elif callable(on_message):
            self._instance.prepare_with_callback(on_message)
        else:
            raise Exception("Message callback is not callable")
        return self
    

    def activate(self):
        """
        Start the pipeline after it is already intialized using prepare()

        Returns:
            Pipeline instance

        Raises:
            None
        """
        self._instance.activate()
        return self
    

    def wait(self):
        """
        Wait until the pipeline is stopped.
        Caller thread releases the GIC before yielding and tries grab it after wakeup

        Returns:
            Pipeline instance

        Raises:
            None
        """
        self._instance.wait()
        return self

    def stop(self):
        """
        Stop the pipeline

        Returns:
            Pipeline instance

        Raises:
            None
        """
        self._instance.stop()
        return self

    def start_rtsp_server(self, rtsp_port, udp_port, buffer_size=0):
        """
        Start a RTSP server
        Args:
            rtsp_port (int): server port
            udp_port  (int): packet transmission port
            buffer_size (int): size of the sender buffer
        Return: server id
        Raises: None
        """
        return self._instance.start_rtsp_server(rtsp_port, udp_port, buffer_size)
    
    def seek(self, time_in_seconds: float):
        """
        Seek to a specific time for the pipeline to begin processing
        Args:
            time_in_seconds: the time to begin with
        """
        self._instance.seek(int(time_in_seconds*1e9))
        return self

    def start_recording(self, source_name: str, start_time: int, duration: int, callback: callable = None):
        """
        Start smart recording on a source.
        Args: source_name, start_time, duration, callback
            source_name: name of the source to record
            start_time: time to start recording
            duration: duration of the recording
            callback: callback function to be called when the recording is done
        Returns: session id
        """
        session_id = self._instance.start_recording(source_name, start_time, duration, callback)
        return session_id

    def stop_recording(self, source_name: str):
        """
        Stop smart recording on a source.
        Args: source_name
        Returns: True if successful, False otherwise
        """
        return self._instance.stop_recording(source_name)

    def stop_recording_by_session_id(self, session_id: int):
        """
        Stop smart recording on a session id.
        Args: session_id
        Returns: True if successful, False otherwise
        """
        return self._instance.stop_recording_by_session_id(session_id)