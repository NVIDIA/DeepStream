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

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd
from multiprocessing import Process
from collections import deque
import os
import sys
import time

PIPELINE_NAME = "deepstream-3d-action-recognition"
FPS_INTERVAL = 30
MAX_STR_LEN = 2048


class FpsCalculation:
  """Calculates current and average FPS per source using a sliding timestamp window."""

  def __init__(self, interval=50):
    self._max_frame_nums = interval
    self._timestamps = {}   # source_id -> deque of timestamps
    self._fps_stats = {}    # source_id -> {start_time, sum_frames, cur_fps, avg_fps}

  def update_fps(self, source_id):
    now = time.time()
    fps = -1.0

    if source_id in self._timestamps:
      tms = self._timestamps[source_id]
      if tms:
        fps = len(tms) / (now - tms[0])
      while len(tms) >= self._max_frame_nums:
        tms.popleft()

      stats = self._fps_stats[source_id]
      stats["cur_fps"] = fps
      stats["avg_fps"] = stats["sum_frames"] / (now - stats["start_time"])
      stats["sum_frames"] += 1
    else:
      self._timestamps[source_id] = deque()
      self._fps_stats[source_id] = {
        "start_time": now,
        "sum_frames": 1,
        "cur_fps": 0.0,
        "avg_fps": 0.0,
      }

    self._timestamps[source_id].append(now)
    return fps

  def get_all_fps(self):
    """Returns a list of (cur_fps, avg_fps) tuples for all sources, ordered by source_id."""
    return [
      (self._fps_stats[sid]["cur_fps"], self._fps_stats[sid]["avg_fps"])
      for sid in sorted(self._fps_stats)
    ]


g_fps_cal = FpsCalculation(50)


class AddLabel(BatchMetadataOperator):
  def __init__(self):
    super().__init__()
    self._frame_count = 0

  def _add_label_text(self, batch_meta, roi_meta, label):
    display_text = "Label: " + label
    text = osd.Text()
    text.display_text = display_text.encode('ascii')
    text.x_offset = int(roi_meta.roi.left)
    text.y_offset = max(int(roi_meta.roi.top - 10), 0)
    text.font.name = osd.FontFamily.Serif
    text.font.size = 12
    text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
    text.set_bg_color = True
    text.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)

    display_meta = batch_meta.acquire_display_meta()
    display_meta.add_text(text)
    roi_meta.frame_meta.append(display_meta)

  def _add_fps_text(self, batch_meta, frame_meta, fps):
    fps_text = "FPS: {:.2f}".format(fps)
    text = osd.Text()
    text.display_text = fps_text.encode('ascii')
    text.x_offset = 0
    text.y_offset = 40
    text.font.name = osd.FontFamily.Serif
    text.font.size = 10
    text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
    text.set_bg_color = True
    text.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)

    display_meta = batch_meta.acquire_display_meta()
    display_meta.add_text(text)
    frame_meta.append(display_meta)

  def handle_metadata(self, batch_meta):
    for user_meta in batch_meta.preprocess_batch_items:
      preprocess_batch_meta = user_meta.as_preprocess_batch()
      if not preprocess_batch_meta:
        continue
      for roi_meta in preprocess_batch_meta.rois:
        for classifier_meta in roi_meta.classifier_items:
          num_labels = classifier_meta.n_labels
          for i in range(num_labels):
            label = classifier_meta.get_n_label(i)
            self._add_label_text(batch_meta, roi_meta, label)

    for frame_meta in batch_meta.frame_items:
      fps = g_fps_cal.update_fps(frame_meta.source_id)
      if fps >= 0:
        self._add_fps_text(batch_meta, frame_meta, fps)

    self._frame_count += 1
    if self._frame_count >= FPS_INTERVAL:
      self._frame_count = 0
      all_fps = g_fps_cal.get_all_fps()
      if all_fps:
        fps_str = "\t".join("{:.2f} ({:.2f})".format(cur, avg) for cur, avg in all_fps)
        print("FPS(cur/avg): {}".format(fps_str))



def main(file_path):
    file_ext = os.path.splitext(file_path)[1]
    if file_ext in [".yaml", ".yml"]:
        pipeline = Pipeline(PIPELINE_NAME, file_path)
        pipeline.attach("pgie", Probe("add_label_to_display", AddLabel())).start().wait()
    
    else:
        print("Invalid File Type: " + file_path + " Please provide a .yaml config file")



if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <YAML config file>\n" % sys.argv[0])
        sys.exit(1)
    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()