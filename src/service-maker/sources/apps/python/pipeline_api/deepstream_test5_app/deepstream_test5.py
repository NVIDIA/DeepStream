#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, PipelineState, StateTransitionMessage, DynamicSourceMessage, Probe, BatchMetadataOperator
from pyservicemaker import osd, signal, utils
from pyservicemaker import CommonFactory, SourceConfig, SensorInfo
from multiprocessing import Process
import sys
import argparse
import platform

PGIE = "pgie"
CHECKPOINT = "queue_tee"
TILER = "tiler"
MSGBROKER = "msgbroker"
MSGCONV = "msgconv"
OSD = "osd"
SINK = "sink"

class EventMessageGenerator(BatchMetadataOperator):
    """
    generate event message user metadata for downstream nvmsgconv
    to produce message payloads, which will be posted to the remote
    server by msgboker, e.g kafka
    """
    def __init__(self, sensor_map, labels):
        super().__init__()
        self._sensor_map = sensor_map
        self._labels = labels

    def handle_metadata(self,  batch_meta, frame_interval=1):
        for frame_meta in batch_meta.frame_items:
            frame_num = frame_meta.frame_number
            for object_meta in frame_meta.object_items:
                if not (frame_num % frame_interval):
                    event_msg = batch_meta.acquire_event_message_meta()
                    if event_msg:
                        source_id = frame_meta.source_id
                        sensor_info = self._sensor_map[source_id] if source_id in self._sensor_map else None
                        if sensor_info != None:
                            sensor_id = sensor_info.sensor_id if sensor_info else "N/A"
                            uri = sensor_info.uri if sensor_info else "N/A"
                            event_msg.generate(object_meta, frame_meta, sensor_id, uri, self._labels)
                            frame_meta.append(event_msg)

def on_message(message, osd_event_handler, sensor_map, perf_monitor, model_engine_monitor):
    if isinstance(message, StateTransitionMessage):
        if message.new_state == PipelineState.PLAYING and message.origin == "sink":
            if not osd_event_handler.started:
                osd_event_handler.start()
            if not model_engine_monitor.started:
                model_engine_monitor.start()
    elif isinstance(message, DynamicSourceMessage):
        source_id = message.source_id
        if message.source_added:
            if source_id in sensor_map:
                print(f"Warning, sensor with source id {source_id} already added")
            else:
                sensor_map[source_id] = SensorInfo(
                    sensor_id=message.sensor_id,
                    sensor_name=message.sensor_name,
                    uri=message.uri)
            perf_monitor.add_stream(
                source_id=source_id,
                sensor_id=message.sensor_id,
                sensor_name=message.sensor_name,
                uri=message.uri)
        else:
            del sensor_map[source_id]
            perf_monitor.remove_stream(source_id)

def main(args):
    # Check input arguments
    if len(args.pipeline_config) != len(args.source_config):
        sys.stderr.write("Number of pipeline config must equal to that of source config")
        sys.exit(1)
    labels = []
    if args.label_file:
        with open(args.label_file, 'r') as f:
            for label in f:
                labels.append(label.strip().lower())

    source_maps = [dict() for _ in args.pipeline_config]
    reference_holders = []
    pipelines = []
    pipeline_id = 0
    for source_config_file, pipeline_config_file in zip(args.source_config, args.pipeline_config):
        sensor_map = source_maps[pipeline_id]
        pipeline_name = f"deepstream-test5-{pipeline_id}"
        pipeline = Pipeline(name=pipeline_name, config_file=pipeline_config_file)
        pipelines.append(pipeline)
        source_config = SourceConfig()
        source_config.load(source_config_file)
        source_properties = dict(source_config.source_properties)
        use_multisrcbin = True if source_config.source_type == "nvmultiurisrcbin" else False
        use_camerabin = True if source_config.source_type == "camerabin" else False
        if use_multisrcbin:
            source_properties["uri-list"] = ','.join([source.uri for source in source_config.sensor_list])
            source_properties["sensor-id-list"] = ','.join([source.sensor_id for source in source_config.sensor_list])
            source_properties["sensor-name-list"] = ','.join([source.sensor_name for source in source_config.sensor_list])
            pipeline.add("nvmultiurisrcbin", "source", source_properties).link("source", PGIE)
        elif use_camerabin:
            streammux_properties = {
                "batch-size": len(source_config.camera_list),
                "width": pipeline[TILER].get("width"),
                "height": pipeline[TILER].get("height"),
                "live-source": 1,
                "batched-push-timeout": 40000
            }
            pipeline.add("nvstreammux", "mux", streammux_properties)
            if "gpu-id" in source_config.source_properties:
                pipeline["mux"].set("gpu-id", source_config.source_properties["gpu-id"])
            for i, camera_info in enumerate(source_config.camera_list):

                invalid_value_check = ""
                if not camera_info.camera_type:
                    invalid_value_check = "camera-type"
                if not camera_info.camera_width:
                    invalid_value_check = "camera-width"
                if not camera_info.camera_height:
                    invalid_value_check = "camera-height"
                if not camera_info.camera_fps_n:
                    invalid_value_check = "camera-fps-n"
                if not camera_info.camera_fps_d:
                    invalid_value_check = "camera-fps-d"

                if len(invalid_value_check):
                    print(f"No Value for {invalid_value_check}")
                    sys.exit()

                src_name = f"src_{i}"
                src_cap_filter = f"{src_name}_cap_filter"
                capfilterCaps = f"video/x-raw(memory:NVMM), format={camera_info.camera_video_format if camera_info.camera_video_format else 'NV12'}, width={camera_info.camera_width}, height={camera_info.camera_height}, framerate={camera_info.camera_fps_n}/{camera_info.camera_fps_d}"
                pipeline.add("capsfilter", src_cap_filter, {"caps": capfilterCaps})
                src_queue_mux = f"{src_name}_queue_mux"
                pipeline.add("queue", src_queue_mux)

                if camera_info.camera_type == "CSI":
                    if (camera_info.camera_csi_sensor_id == None):
                        print("camera-csi-sensor-id required")
                        sys.exit()
                    else:
                        pipeline.add("nvarguscamerasrc" if platform.processor() == "aarch64" else "videotestsrc", src_name, {"sensor-id": camera_info.camera_csi_sensor_id})
                        pipeline.link(src_name, src_cap_filter, src_queue_mux, "mux")
                        sensor_map[i] = SensorInfo(camera_info.camera_csi_sensor_id, f"{i}", src_name)

                elif camera_info.camera_type == "V4L2":
                    if (camera_info.camera_v4l2_dev_node == None):
                        print("camera-v4l2-dev-node required")
                        sys.exit()
                    else:
                        device = f"/dev/video{camera_info.camera_v4l2_dev_node}"
                        pipeline.add("v4l2src", src_name, {"device": device})

                        src_cap_filter1 = f"{src_name}_cap_filter1"
                        capfilterCaps1 = f"video/x-raw, width={camera_info.camera_width}, height={camera_info.camera_height}, framerate={camera_info.camera_fps_n}/{camera_info.camera_fps_d}"
                        pipeline.add("capsfilter", src_cap_filter1, {"caps": capfilterCaps1})

                        nvvidconv2 = f"{src_name}_nvvidconv2"

                        if platform.processor() == "aarch64" and camera_info.nvvideoconvert_copy_hw:
                            pipeline.add("nvvideoconvert", nvvidconv2, {"gpu-id": camera_info.gpu_id if camera_info.gpu_id else 0,
                                                                        "nvbuf-memory-type": camera_info.nvbuf_mem_type if camera_info.nvbuf_mem_type else 0,
                                                                        "copy-hw": camera_info.nvvideoconvert_copy_hw})

                        else:
                            pipeline.add("nvvideoconvert", nvvidconv2, {"gpu-id": camera_info.gpu_id if camera_info.gpu_id else 0,
                                                                        "nvbuf-memory-type": camera_info.nvbuf_mem_type if camera_info.nvbuf_mem_type else 0})
                        pipeline.link(src_name, src_cap_filter1)

                        if platform.processor() != "aarch64":
                            nvvidconv1 = f"{src_name}_nvvidconv1"
                            pipeline.add("videoconvert", nvvidconv1)
                            pipeline.link(src_cap_filter1, nvvidconv1, nvvidconv2)
                        else:
                            pipeline.link(src_cap_filter1, nvvidconv2)

                        pipeline.link(nvvidconv2, src_cap_filter, src_queue_mux, "mux")
                        sensor_map[i] = SensorInfo(device, f"{i}", src_name)
                else:
                    print("Invalid camera-type value")
                    sys.exit()

            pipeline.add("queue", "queue_pgie")
            pipeline.link("mux", "queue_pgie", PGIE)

        else:
            streammux_properties = {
                "batch-size": len(source_config.sensor_list),
                "width": pipeline[TILER].get("width"),
                "height": pipeline[TILER].get("height")
            }
            pipeline.add("nvstreammux", "mux", streammux_properties)
            if "gpu-id" in source_config.source_properties:
                pipeline["mux"].set("gpu-id", source_config.source_properties["gpu-id"])
            for i, sensor_info in enumerate(source_config.sensor_list):
                source_properties["uri"] = sensor_info.uri
                src_name = f"src_{i}"
                pipeline.add(source_config.source_type, src_name, source_properties).link((src_name, "mux"), ("vsrc_%u", ""))
                sensor_map[i] = sensor_info
            pipeline.link("mux", PGIE)
            sr_controller = CommonFactory.create("smart_recording_action", "sr_controller")
            reference_holders.append(sr_controller)

            if isinstance(sr_controller, signal.Emitter):
                sr_controller_properties = {
                        "proto-lib": pipeline[MSGBROKER].get("proto-lib"),
                        "conn-str": pipeline[MSGBROKER].get("conn-str"),
                        "msgconv-config-file": pipeline[MSGCONV].get("config"),
                        "proto-config-file": pipeline[MSGBROKER].get("config"),
                        "topic-list": "test5-sr"
                }
                sr_controller.set(sr_controller_properties)
                for i, _ in enumerate(source_config.sensor_list):
                    source_name = f"src_{i}"
                    sr_controller.attach("start-sr", pipeline[source_name])
                    sr_controller.attach("stop-sr", pipeline[source_name])
                    pipeline.attach(source_name, "smart_recording_signal", "sr", "sr-done")
            else:
                print("Warning: failed to create smart recording controller")

        pipeline.attach(CHECKPOINT, Probe("message_generator", EventMessageGenerator(sensor_map, labels)))

        # OSD mouse event handling
        osd_event_handler = osd.EventHandler(osd=pipeline[OSD], tiler=pipeline[TILER], renderer=pipeline[SINK])
        reference_holders.append(osd_event_handler)
        if platform.processor() != "aarch64":
            osd_event_handler.start()

        # fps measurement
        perf_monitor = utils.PerfMonitor(
            batch_size=len(source_config.camera_list) if use_camerabin else len(source_config.sensor_list),
            interval=args.perf_measurement_interval_sec,
            source_type=source_config.source_type,
            show_name= not args.hide_stream_name)
        perf_monitor.apply(pipeline[TILER], "sink")
        reference_holders.append(perf_monitor)

        # model engine file monitor
        model_engine_monitor = utils.EngineFileMonitor(pipeline[PGIE], pipeline[PGIE].get("model-engine-file"))
        reference_holders.append(model_engine_monitor)
        pipeline.prepare(lambda m, o=osd_event_handler, s=sensor_map, p=perf_monitor, mem=model_engine_monitor: on_message(m, o, s, p, mem))
        pipeline_id += 1
        if platform.processor() == "aarch64":
            osd_event_handler.start()

    for pipeline in pipelines:
        pipeline.activate()

    for pipeline in pipelines:
        pipeline.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepstream Test5')
    parser.add_argument('-c', '--pipeline-config', nargs='+', required=True, help='YAML configuration file for a pipeline')
    parser.add_argument('-s', '--source-config', nargs='+', required=True, help='YAML configuration file for sources of a pipeline')
    parser.add_argument('-l', '--label-file', nargs='*', help='Label file for custom models')
    parser.add_argument('--perf-measurement-interval-sec', type=int, default=1, help='Interval of performance measurement in seconds')
    parser.add_argument('--hide-stream-name', action='store_true', default=False, help="Don't show stream name in fps measurement data")
    args = parser.parse_args()
    
    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main,  args=(args,))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()