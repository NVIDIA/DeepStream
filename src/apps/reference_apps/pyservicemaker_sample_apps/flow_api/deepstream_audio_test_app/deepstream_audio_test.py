###################################################################################################
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
###################################################################################################

import argparse
import os
from pathlib import Path

from pyservicemaker import BatchMetadataOperator, Flow, MediaType, Pipeline, Probe, RenderMode


AUDIO_FILE = "/opt/nvidia/deepstream/deepstream/samples/streams/sonyc_mixed_audio.wav"
SONYC_INFER_CONFIG = (
    "config_infer_audio_sonyc.txt"
)
SONYC_AUDIO_TRANSFORM = (
    "melsdb,fft_length=2560,hop_size=692,dsp_window=hann,num_mels=128,"
    "sample_rate=44100,p2db_ref=(float)1.0,p2db_min_power=(float)0.0,"
    "p2db_top_db=(float)80.0"
)
SONYC_AUDIO_RATE = 44100
SONYC_AUDIO_FRAME_SIZE = 441000
SONYC_AUDIO_HOP_SIZE = 110250


class AudioClassificationPrinter(BatchMetadataOperator):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
        self.batches = 0

    def handle_metadata(self, batch_meta):
        self.batches += 1
        if self.interval <= 0 or self.batches % self.interval != 0:
            return

        labels = []
        for metadata in batch_meta.extract(max(1, batch_meta.n_frames)):
            if not metadata:
                continue
            for label_group in metadata.get("labels", []):
                labels.extend(label_group)

        summary = ", ".join(labels) if labels else "no labels above threshold"
        print(f"Audio inference batch {self.batches}: {summary}")


def run_audio_infer(args):
    if os.environ.get("USE_NEW_NVSTREAMMUX", "").lower() not in {"yes", "1", "true"}:
        raise RuntimeError("Infer mode requires nvstreammux2. Set USE_NEW_NVSTREAMMUX=yes in the environment.")

    inputs = args.inputs or [AUDIO_FILE, AUDIO_FILE]
    batch_size = args.batch_size or len(inputs)
    capture_kwargs = {"file_loop": True} if args.file_loop else {}

    flow = Flow(Pipeline("deepstream-audio")).capture(
        inputs,
        media_types=MediaType.AUDIO,
        **capture_kwargs
    ).batch(batch_size=batch_size)

    audio_caps = f"audio/x-raw, rate={args.audio_input_rate}"
    flow = flow.infer(
        args.infer_config,
        batch_size=batch_size,
        audio_framesize=args.audio_framesize,
        audio_hopsize=args.audio_hopsize,
        audio_transform=args.audio_transform,
        audio_caps=audio_caps,
    )

    if args.print_interval:
        flow = flow.attach(Probe("audio-classification-printer", AudioClassificationPrinter(args.print_interval)))
    flow.render(RenderMode.DISCARD, sync=False)()


def run_audio_io(args):
    inputs = args.inputs or [AUDIO_FILE]
    if len(inputs) != 1:
        raise ValueError(f"{args.mode} mode expects exactly one input")

    capture_kwargs = {"file_loop": True} if args.file_loop else {}
    flow = Flow(Pipeline("deepstream-audio-test")).capture(
        inputs,
        media_types=MediaType.AUDIO,
        **capture_kwargs
    )
    if args.mode == "encode":
        flow.encode(args.output)()
        return
    if args.mode == "stream":
        flow.render(RenderMode.STREAM, rtsp_port=args.rtsp_port, rtsp_mount_point=args.rtsp_mount_point, sync=False)()
        return

    mode = RenderMode.DISCARD if args.mode == "discard" else RenderMode.DISPLAY
    flow.render(mode, audio_sink=args.audio_sink)()


def main():
    parser = argparse.ArgumentParser(description="PyServiceMaker audio Flow API sample")
    parser.add_argument("inputs", nargs="*", help="audio source URI or file path")
    parser.add_argument("--mode", choices=["infer", "discard", "play", "stream", "encode"], default="infer")
    parser.add_argument("--output", default=str(Path.home() / "sonyc_copy.wav"), help="output WAV path for encode mode")
    parser.add_argument("--audio-sink", default="autoaudiosink", help="audio sink element for play mode")
    parser.add_argument("--rtsp-port", type=int, default=8554, help="RTSP port for stream mode")
    parser.add_argument("--rtsp-mount-point", default="ds-audio", help="RTSP mount point for stream mode")
    parser.add_argument("--infer-config", default=SONYC_INFER_CONFIG, help="nvinferaudio config file")
    parser.add_argument("--batch-size", type=int, help="audio batch size; defaults to number of inputs")
    parser.add_argument("--audio-input-rate", type=int, default=SONYC_AUDIO_RATE, help="audio rate before nvinferaudio")
    parser.add_argument("--audio-framesize", type=int, default=SONYC_AUDIO_FRAME_SIZE, help="nvinferaudio frame size")
    parser.add_argument("--audio-hopsize", type=int, default=SONYC_AUDIO_HOP_SIZE, help="nvinferaudio hop size")
    parser.add_argument("--audio-transform", default=SONYC_AUDIO_TRANSFORM, help="nvinferaudio audio-transform structure")
    parser.add_argument("--file-loop", action="store_true", help="loop file inputs, matching the legacy deepstream-audio test config")
    parser.add_argument("--print-interval", type=int, default=1, help="print classifier labels every N inference batches; 0 disables printing")
    args = parser.parse_args()

    if args.mode == "infer":
        run_audio_infer(args)
        return

    run_audio_io(args)


if __name__ == "__main__":
    main()
