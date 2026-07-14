#!/usr/bin/env python3
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


import os, tkinter as tk, argparse
from datetime import datetime
from kafka import KafkaConsumer
from google.protobuf.json_format import MessageToDict
from schema_pb2 import Frame
from collections import defaultdict
import numpy as np, yaml, cv2, time
from tqdm import tqdm

def extract_expected_sensors(msgconv_config):
    """Extract expected sensor IDs from config_msgconv.txt file."""
    try:
        expected_sensors = []
        
        with open(msgconv_config, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Check for section header [sensorX]
            if line.startswith('[sensor') and line.endswith(']'):
                current_section = line[1:-1]  # Remove brackets
                continue
                
            # Look for id= lines within sensor sections
            if current_section and line.startswith('id='):
                sensor_id = line.split('=', 1)[1].strip()
                if sensor_id:
                    expected_sensors.append(sensor_id)
        
        if not expected_sensors:
            raise ValueError("No sensor IDs found in msgconv config file")
            
        return expected_sensors
        
    except (FileNotFoundError, ValueError) as e:
        raise Exception(f"Could not parse msgconv config file {msgconv_config}: {e}")
        

class FrameBuffer:
    def __init__(self, expected_sensors=None, timeout=0.5, lookahead_frames=1):
        """
        For a frame t to be considered complete, 1 of the 3 conditions must be met:
            1. Messages from all expected sensors are received at frame t
            2. Current time - message sent time of any sensor at frame t > timeout
            3. At least 1 of the future frames (t+lookahead_frames) are received

        The current settings is designed for single-container MV3DT.
        For multi-container MV3DT, consider increasing the timeout and lookahead_frames.
        """

        self.frame_data = defaultdict(dict)
        self.expected_sensors = expected_sensors or set()
        self.timeout = timeout
        self.timestamps = {}
        self.lookahead_frames = lookahead_frames
        
    def add_frame(self, frame_id, sensor_id, frame_dict):
        try:
            frame_id = int(frame_id)
        except ValueError:
            pass
        if frame_id not in self.timestamps:
            self.timestamps[frame_id] = time.time()
        self.frame_data[frame_id][sensor_id] = frame_dict
        if not self.expected_sensors:
            self.expected_sensors.add(sensor_id)
    
    def get_complete_frame(self):
        current_time = time.time()
        num_expected = len(self.expected_sensors)
        # Process frames in ascending order when possible
        def _key_fn(x):
            try:
                return int(x)
            except Exception:
                return float('inf')
        for frame_id in sorted(list(self.frame_data.keys()), key=_key_fn):
            sensors = set(self.frame_data[frame_id].keys())
            frame_time = self.timestamps.get(frame_id, current_time)
            # Condition 1: full set received for this frame
            full_received = (num_expected > 0 and len(sensors) >= num_expected and sensors.issubset(self.expected_sensors))
            # Condition 2: timeout
            timed_out = (current_time - frame_time) > self.timeout
            # Condition 3: at least 1 of future frame (t+lookahead_frames) received
            future_received = False
            if num_expected > 0:
                try:
                    base_id = int(frame_id)
                    future_id = base_id + self.lookahead_frames
                    future_data = self.frame_data.get(future_id)
                    if future_data is not None:
                        future_received = True
                except Exception:
                    pass
            if full_received or timed_out or future_received:
                # print ('frame_id', frame_id, 'full_received', full_received, 'timed_out', timed_out, 'future_received', future_received)
                frame_data = self.frame_data.pop(frame_id)
                self.timestamps.pop(frame_id, None)
                return frame_id, frame_data
        return None, None
    
    def get_all_complete_frames(self):
        complete_frames = []
        for frame_id, sensors_data in self.frame_data.items():
            if len(sensors_data) > 0:
                try:
                    complete_frames.append((int(frame_id), frame_id, sensors_data))
                except (ValueError, TypeError):
                    complete_frames.append((0, frame_id, sensors_data))
        complete_frames.sort(key=lambda x: x[0])
        return [(frame_id, data) for _, frame_id, data in complete_frames]
    
    def get_frame(self, frame_id):
        """Get frame data for specific frame_id without waiting for conditions, and remove it from buffer"""
        if frame_id in self.frame_data:
            frame_data = self.frame_data.pop(frame_id)
            self.timestamps.pop(frame_id, None)
            return frame_id, frame_data
        return None, None

def display_frame(vis_img, frame_id, video_writer):
    """Add frame info, display frame, and write to video"""
    recording = "REC" if video_writer else ""
    info = f"Frame: {frame_id} {recording}"
    cv2.putText(vis_img, info, (10, vis_img.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imshow('Bird-Eye View of Multi-View 3D Tracking', vis_img)

    if video_writer:
        video_writer.write(vis_img)

def load_map_and_transforms(dataset_path):
    """Load map image and transformation matrix from dataset"""
    map_path = os.path.join(dataset_path, 'map.png')
    transforms_path = os.path.join(dataset_path, 'transforms.yml')

    with open(transforms_path, 'r') as f:
        transforms = yaml.safe_load(f)
    T_ov2px = np.array(transforms['T_ov2px']).reshape(3, 3)

    map_img = cv2.imread(map_path)
    if map_img is None:
        raise FileNotFoundError(f"Map image not found at {map_path}")
    
    return map_img, T_ov2px

def setup_map_scaling(map_img, target_width_ratio=0.8, target_height_ratio=0.8, allow_headless=False):
    """Setup map scaling based on screen size"""
    try:
        root = tk.Tk()
        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except tk.TclError:
        if not allow_headless:
            raise
        screen_width = int(os.environ.get('MV3DT_BEV_WIDTH', 1280))
        screen_height = int(os.environ.get('MV3DT_BEV_HEIGHT', 720))

    target_width = int(screen_width * target_width_ratio)
    target_height = int(screen_height * target_height_ratio)

    map_height, map_width = map_img.shape[:2]
    scale = min(target_width / map_width, target_height / map_height)
    new_width, new_height = int(map_width * scale), int(map_height * scale)

    map_img_resized = cv2.resize(map_img, (new_width, new_height))
    scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

    return map_img_resized, scale_matrix, new_width, new_height

def create_kafka_consumer(group_id, consumer_timeout_ms=None, broker='localhost:9092',
                          topic='mv3dt', auto_offset_reset='earliest', from_end=False):
    """Create and configure Kafka consumer"""
    config = {
        'bootstrap_servers': broker,
        'auto_offset_reset': auto_offset_reset,
        'value_deserializer': lambda x: x,
        'group_id': group_id,
        'enable_auto_commit': True
    }

    if consumer_timeout_ms is not None:
        config['consumer_timeout_ms'] = consumer_timeout_ms

    consumer = KafkaConsumer(**config)
    consumer.subscribe([topic])
    if broker == 'localhost:9092' and topic == 'mv3dt':
        print("Connected to Kafka and subscribed to 'mv3dt' topic")
    else:
        print(f"Connected to Kafka broker '{broker}' and subscribed to '{topic}' topic")

    if from_end:
        assignment_deadline = time.time() + 10
        while not consumer.assignment() and time.time() < assignment_deadline:
            consumer.poll(timeout_ms=100)
            time.sleep(0.1)
        if consumer.assignment():
            consumer.seek_to_end()
            print(f"Seeking to current end offsets for topic '{topic}'")
        else:
            print(f"Warning: no partitions assigned for topic '{topic}' before seek-to-end")

    return consumer

def draw_objects_on_map(frame_data, T_ov2px, map_img, trajectories, object_colors, frame_id, frame_history, show_ids=False, average_multi_cam=False):
    vis_img = map_img.copy()
    colors = [
        (255, 0, 0), (0, 255, 255), (139, 69, 19), (0, 255, 0), (255, 0, 255),
        (50, 205, 50), (255, 140, 0), (0, 0, 255), (255, 165, 0), (255, 105, 180),
        (75, 0, 130), (255, 255, 0), (0, 128, 128), (0, 191, 255), (154, 205, 50),
        (255, 20, 147), (30, 144, 255), (128, 0, 128), (220, 20, 60), (0, 206, 209)
    ]
    
    all_objects = [obj for frame_dict in frame_data.values() for obj in frame_dict.get('objects', [])]
    
    try:
        current_frame_num = int(frame_id)
        frame_history.append(current_frame_num)
        if len(frame_history) > 240:
            frame_history = frame_history[-240:]
    except:
        current_frame_num = 0
    
    current_objects = set()
    
    if average_multi_cam:
        # Group objects by ID across all cameras for averaging
        objects_by_id = defaultdict(list)
        for obj in all_objects:
            bbox_3d = obj.get('bbox3d', {}).get('coordinates', {})
            if bbox_3d:
                try:
                    world_x, world_y = bbox_3d[:2]
                    object_id = obj.get('id', 0)
                    objects_by_id[object_id].append((world_x, world_y))
                except:
                    continue
        
        # Calculate average positions and add to trajectories
        for object_id, positions in objects_by_id.items():
            if positions:
                # Calculate average world position
                avg_world_x = sum(pos[0] for pos in positions) / len(positions)
                avg_world_y = sum(pos[1] for pos in positions) / len(positions)
                
                try:
                    # Convert to pixel coordinates
                    pt_ov_h = np.array([avg_world_x, avg_world_y, 1.0])
                    pt_px_h = np.dot(T_ov2px, pt_ov_h)
                    pt_px_h /= pt_px_h[2]
                    px_x, px_y = int(pt_px_h[0]), int(pt_px_h[1])
                    
                    current_objects.add(object_id)
                    
                    if object_id not in object_colors:
                        object_colors[object_id] = colors[len(object_colors) % len(colors)]
                    
                    trajectories[object_id].append((px_x, px_y, current_frame_num))
                except:
                    continue
    else:
        # Original behavior: show all trajectory points from all cameras
        for obj in all_objects:
            bbox_3d = obj.get('bbox3d', {}).get('coordinates', {})
            if not bbox_3d:
                continue
            try:
                world_x, world_y = bbox_3d[:2]
                pt_ov_h = np.array([world_x, world_y, 1.0])
                pt_px_h = np.dot(T_ov2px, pt_ov_h)
                pt_px_h /= pt_px_h[2]
                px_x, px_y = int(pt_px_h[0]), int(pt_px_h[1])
                
                object_id = obj.get('id', 0)
                current_objects.add(object_id)
                
                if object_id not in object_colors:
                    object_colors[object_id] = colors[len(object_colors) % len(colors)]
                
                trajectories[object_id].append((px_x, px_y, current_frame_num))
            except:
                continue
    
    # Cleanup old trajectory points
    frame_threshold = current_frame_num - 240
    for object_id in list(trajectories.keys()):
        trajectories[object_id] = [(x, y, f) for x, y, f in trajectories[object_id] if f >= frame_threshold]
        if not trajectories[object_id]:
            del trajectories[object_id]
            object_colors.pop(object_id, None)
    
    # Draw trajectories
    for object_id, traj_points in trajectories.items():
        if not traj_points:
            continue
        color = object_colors.get(object_id, (128, 128, 128))
        base_alpha = 0.9 if object_id in current_objects else 0.6
        min_alpha = 0.3  # Minimum brightness to prevent complete black
        
        for i, (x, y, _) in enumerate(traj_points):
            # Slower fade: use square root for gentler curve
            fade_ratio = (i / max(1, len(traj_points) - 1)) ** 0.5
            fade = min_alpha + (base_alpha - min_alpha) * fade_ratio
            fade_color = tuple(int(c * fade) for c in color)
            cv2.circle(vis_img, (x, y), 1, fade_color, -1)
    
    # Draw current positions with ID labels
    if average_multi_cam:
        # Draw averaged positions for each object
        objects_by_id = defaultdict(list)
        for obj in all_objects:
            bbox_3d = obj.get('bbox3d', {}).get('coordinates', {})
            if bbox_3d:
                try:
                    world_x, world_y = bbox_3d[:2]
                    object_id = obj.get('id', 0)
                    objects_by_id[object_id].append((world_x, world_y))
                except:
                    continue
        
        for object_id, positions in objects_by_id.items():
            if positions and object_id in object_colors:
                # Calculate average world position
                avg_world_x = sum(pos[0] for pos in positions) / len(positions)
                avg_world_y = sum(pos[1] for pos in positions) / len(positions)
                
                try:
                    # Convert to pixel coordinates
                    pt_ov_h = np.array([avg_world_x, avg_world_y, 1.0])
                    pt_px_h = np.dot(T_ov2px, pt_ov_h)
                    pt_px_h /= pt_px_h[2]
                    px_x, px_y = int(pt_px_h[0]), int(pt_px_h[1])
                    
                    # Draw object circle
                    cv2.circle(vis_img, (px_x, px_y), 3, object_colors[object_id], -1)
                    
                    # Draw ID label near the object (if enabled)
                    if show_ids:
                        label_x = px_x + 8
                        label_y = px_y - 8
                        cv2.putText(vis_img, str(object_id), (label_x, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                except:
                    continue
    else:
        # Original behavior: draw all object positions from all cameras
        for obj in all_objects:
            bbox_3d = obj.get('bbox3d', {}).get('coordinates', {})
            if bbox_3d:
                try:
                    world_x, world_y = bbox_3d[:2]
                    pt_ov_h = np.array([world_x, world_y, 1.0])
                    pt_px_h = np.dot(T_ov2px, pt_ov_h)
                    pt_px_h /= pt_px_h[2]
                    px_x, px_y = int(pt_px_h[0]), int(pt_px_h[1])
                    
                    object_id = obj.get('id', 0)
                    if object_id in object_colors:
                        # Draw object circle
                        cv2.circle(vis_img, (px_x, px_y), 3, object_colors[object_id], -1)
                        
                        # Draw ID label near the object (if enabled)
                        if show_ids:
                            label_x = px_x + 8
                            label_y = px_y - 8
                            cv2.putText(vis_img, str(object_id), (label_x, label_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                except:
                    continue
    
    return vis_img, frame_history

def collect_all_messages(consumer, expected_sensors, max_timeout=300,
                         first_message_timeout=10, idle_timeout=10,
                         skip_first_frame_filter=False, verbose=False):
    frame_buffer = FrameBuffer(expected_sensors=expected_sensors)
    start_time = time.time()
    last_msg_time = start_time
    count = 0
    first_frame = True
    
    consumer._consumer_timeout_ms = 1000
    
    while True:
        current_time = time.time()
        if current_time - start_time > max_timeout:
            break
        if count == 0:
            if current_time - start_time > first_message_timeout:
                break
        elif current_time - last_msg_time > idle_timeout:
            break
        
        batch = consumer.poll(timeout_ms=1000)
        if not batch:
            continue
            
        for _, messages in batch.items():
            for msg in messages:
                try:
                    frame = Frame()
                    frame.ParseFromString(msg.value)
                    frame_dict = MessageToDict(frame)
                    frame_id = frame_dict.get('id', 'unknown')
                    
                    if first_frame and frame_id != 'unknown':
                        if not skip_first_frame_filter:
                            try:
                                if int(frame_id) > 100:
                                    if verbose:
                                        print(f"Discarding frame {frame_id} (from previous run)")
                                    first_frame = False
                                    continue
                            except (ValueError, TypeError):
                                pass
                        first_frame = False

                    frame_buffer.add_frame(frame_id, frame_dict.get('sensorId', 'unknown'), frame_dict)
                    count += 1
                    last_msg_time = current_time
                except:
                    continue
    
    frame_buffer.message_count = count
    print(f"Collected {count} messages")
    return frame_buffer

def generate_video(dataset_path, output_path, show_ids, expected_sensors, average_multi_cam,
                   broker='localhost:9092', topic='mv3dt', from_end=False,
                   first_message_timeout=10, idle_timeout=10, max_timeout=300, verbose=False):
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_output_path = os.path.join(output_path, f"trajectory_video_{timestamp}.mp4")
    
    # Setup common components
    map_img, T_ov2px = load_map_and_transforms(dataset_path)
    map_img_resized, scale_matrix, new_width, new_height = setup_map_scaling(map_img, allow_headless=True)
    T_ov2px_scaled = np.dot(scale_matrix, T_ov2px)
    
    try:
        group_id = f'mv3dt_bev_video_{int(time.time())}' if from_end else 'mv3dt_bev_video'
        consumer = create_kafka_consumer(
            group_id,
            broker=broker,
            topic=topic,
            from_end=from_end,
        )
        
        frame_buffer = collect_all_messages(
            consumer,
            expected_sensors,
            max_timeout=max_timeout,
            first_message_timeout=first_message_timeout,
            idle_timeout=idle_timeout,
            skip_first_frame_filter=from_end,
            verbose=verbose,
        )
        complete_frames = frame_buffer.get_all_complete_frames()
        if verbose:
            print(f"Complete frames ready for rendering: {len(complete_frames)}")
        elif not from_end:
            print(frame_buffer.frame_data)
        
        if not complete_frames:
            print(f"No complete frames collected; BEV video was not generated: {video_output_path}")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, 30, (new_width, new_height))
        if not video_writer.isOpened():
            print(f"Error: failed to open video writer for {video_output_path}")
            return
        
        trajectories, object_colors, frame_history = defaultdict(list), {}, []
        
        for frame_id, frame_data in tqdm(complete_frames):
            vis_img, frame_history = draw_objects_on_map(
                frame_data, T_ov2px_scaled, map_img_resized, trajectories, object_colors, frame_id, frame_history, show_ids, average_multi_cam)
            
            cv2.putText(vis_img, f"Frame: {frame_id}", 
                       (10, vis_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            video_writer.write(vis_img)
        
        video_writer.release()
        print(f"Video saved: {video_output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'consumer' in locals():
            consumer.close()

def real_time_visualization(dataset_path, output_path, show_ids, expected_sensors, average_multi_cam, verbose=False):
    # Setup common components
    map_img, T_ov2px = load_map_and_transforms(dataset_path)
    map_img_resized, scale_matrix, new_width, new_height = setup_map_scaling(map_img)
    T_ov2px_scaled = np.dot(scale_matrix, T_ov2px)
    
    frame_buffer = FrameBuffer(expected_sensors=expected_sensors)
    trajectories, object_colors, frame_history = defaultdict(list), {}, []
    
    cv2.namedWindow('Bird-Eye View of Multi-View 3D Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Bird-Eye View of Multi-View 3D Tracking', new_width, new_height)
    
    # Display blank map initially
    initial_img = map_img_resized.copy()
    cv2.imshow('Bird-Eye View of Multi-View 3D Tracking', initial_img)
    cv2.waitKey(1)  # Process window events

    consumer = None
    video_writer = None
    
    try:
        consumer = create_kafka_consumer('mv3dt_visualizer', consumer_timeout_ms=50)
        
        # Wait for partition assignment (up to 5 seconds)
        assignment_timeout = 5.0
        start_time = time.time()
        while not consumer.assignment() and (time.time() - start_time) < assignment_timeout:
            consumer.poll(timeout_ms=100)  # This triggers partition assignment
            time.sleep(0.1)
        
        print(f"Consumer assignment: {consumer.assignment()}")
        if not consumer.assignment():
            print("Warning: No partitions assigned. Topic might not exist or have no partitions.")
            
            # Try to get topic metadata for debugging
            try:
                metadata = consumer.list_consumer_group_offsets()
                topics = consumer.topics()
                print(f"Available topics: {topics}")
                if 'mv3dt' in topics:
                    partitions = consumer.partitions_for_topic('mv3dt')
                    print(f"Partitions for 'mv3dt' topic: {partitions}")
                else:
                    print("Topic 'mv3dt' not found!")
            except Exception as debug_e:
                print(f"Could not get topic metadata: {debug_e}")
            
    except Exception as e:
        print(f"Kafka connection failed: {e}")
    
    print("Controls: 'q'-quit, 'c'-clear, 'r'-record")
    
    last_update = time.time()
    last_frame_id = None
    try:
        while True:
            current_time = time.time()
            
            if consumer:
                try:
                    batch = consumer.poll(timeout_ms=10)
                    for _, messages in batch.items():
                        for msg in messages:
                            try:
                                frame = Frame()
                                frame.ParseFromString(msg.value)
                                frame_dict = MessageToDict(frame)
                                # print ('Received frame', frame_dict.get('id', 'unknown'), 'from sensor', frame_dict.get('sensorId', 'unknown'))
                                frame_buffer.add_frame(frame_dict.get('id', 'unknown'),
                                                     frame_dict.get('sensorId', 'unknown'), frame_dict)
                            except:
                                continue
                except:
                    pass
            
            while True:
                frame_id, frame_data = frame_buffer.get_complete_frame()
                if not frame_data:
                    break
                            
                # If the first frame is > 100, it is likely from a previous run.
                if last_frame_id is None and frame_id > 100:
                    if verbose:
                        print(f"Discarding frame {frame_id} (from previous run)")
                    continue
                
                if last_frame_id is not None and frame_id - last_frame_id > 1:
                    # Process missing frames in between
                    for missing_frame_id in range(last_frame_id + 1, frame_id):
                        missing_id, missing_data = frame_buffer.get_frame(missing_frame_id)
                        if missing_data:
                            # Render frame with tracking data
                            vis_img, frame_history = draw_objects_on_map(missing_data, T_ov2px_scaled, 
                                                                        map_img_resized, trajectories, object_colors, missing_frame_id, frame_history, show_ids, average_multi_cam)
                            display_frame(vis_img, missing_frame_id, video_writer)
                        else:
                            # Render empty frame with correct frame_id
                            empty_vis_img = map_img_resized.copy()
                            display_frame(empty_vis_img, missing_frame_id, video_writer)    
                if last_frame_id is not None and 0 < last_frame_id - frame_id < 30:
                    if verbose:
                        print(f"Received late message from frame {frame_id}, last_frame_id was {last_frame_id}")
                        print(f"Discarding frame {frame_id} (late message)")
                    continue
                last_frame_id = frame_id
                vis_img, frame_history = draw_objects_on_map(frame_data, T_ov2px_scaled, 
                                                            map_img_resized, trajectories, object_colors, frame_id, frame_history, show_ids, average_multi_cam)
                
                display_frame(vis_img, frame_id, video_writer)
            
            last_update = current_time
            
            key = cv2.waitKey(1) & 0xFF
            try:
                window_closed = cv2.getWindowProperty('Bird-Eye View of Multi-View 3D Tracking', cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                window_closed = True
            if key == ord('q') or window_closed:
                break
            elif key == ord('c'):
                trajectories.clear()
                object_colors.clear()
                frame_history.clear()
                print("Cleared trajectories")
            elif key == ord('r'):
                if video_writer is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    live_video_path = os.path.join(output_path, f"live_trajectory_{timestamp}.mp4")
                    os.makedirs(output_path, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(live_video_path, fourcc, 30, (new_width, new_height))
                    print(f"Started recording: {live_video_path}" if video_writer.isOpened() else "Recording failed")
                else:
                    video_writer.release()
                    video_writer = None
                    print("Stopped recording")
                
    finally:
        if video_writer:
            video_writer.release()
        if consumer:
            consumer.close()
        cv2.destroyAllWindows()
        
def parse_args():
    parser = argparse.ArgumentParser(description='Kafka BEV Online Visualizer')
    parser.add_argument('--dataset-path', type=str, 
                       default="datasets/mtmc_4cam",
                       help='Path to dataset)')
    parser.add_argument('--msgconv-config', type=str, 
                       default='config_msgconv.txt',
                       help='Path to message converter config file (config_msgconv.txt)')
    parser.add_argument('--output-path', type=str, 
                       default='output_videos',
                       help='Output directory for videos')
    parser.add_argument('--offline', action='store_true',
                       help='Run in offline mode and save a BEV video from Kafka messages. \
                           Use --from-end when starting before a fresh MV3DT run.')
    parser.add_argument('--broker', type=str, default='localhost:9092',
                       help='Kafka broker server for offline recording')
    parser.add_argument('--topic', type=str, default='mv3dt',
                       help='Kafka topic to consume for offline recording')
    parser.add_argument('--from-end', action='store_true',
                       help='Start offline recording at the current topic end offset before a fresh run')
    parser.add_argument('--first-message-timeout', type=float, default=10,
                       help='Seconds to wait for the first Kafka message in offline mode. Default preserves previous behavior; increase when using --from-end before DeepStream startup')
    parser.add_argument('--idle-timeout', type=float, default=10,
                       help='Seconds to wait after the latest Kafka message before finishing offline capture')
    parser.add_argument('--max-timeout', type=float, default=300,
                       help='Maximum seconds to collect Kafka messages in offline mode')
    parser.add_argument('--show-ids', action='store_true',
                       help='Show object IDs near trajectory heads')
    parser.add_argument('--average-multi-cam', action='store_true',
                       help='Average trajectory points from multiple cameras for the same object')
    parser.add_argument('--verbose', action='store_true',
                       help='Print warnings and diagnostic messages')
    
    return parser.parse_args()

def main():
    args = parse_args()
    expected_sensors = extract_expected_sensors(args.msgconv_config)
    print(f"Expected sensors: {expected_sensors}")
    if args.offline:
        generate_video(
            args.dataset_path,
            args.output_path,
            args.show_ids,
            expected_sensors,
            args.average_multi_cam,
            broker=args.broker,
            topic=args.topic,
            from_end=args.from_end,
            first_message_timeout=args.first_message_timeout,
            idle_timeout=args.idle_timeout,
            max_timeout=args.max_timeout,
            verbose=args.verbose,
        )
    else:
        real_time_visualization(
            args.dataset_path,
            args.output_path,
            args.show_ids,
            expected_sensors,
            args.average_multi_cam,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
