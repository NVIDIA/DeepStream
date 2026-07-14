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
import sys
from multiprocessing import Process
from urllib.parse import urlparse

from pyservicemaker import BatchMetadataOperator, Pipeline, Probe


PIPELINE_NAME = "deepstream-audio"
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


def nvmm_audio_caps(rate):
    return (
        f"audio/x-raw(memory:NVMM), rate={rate}, "
        "format=S16LE, layout=interleaved, channels=1"
    )


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


def normalize_uri(input_uri):
    parsed = urlparse(input_uri)
    if parsed.scheme and parsed.scheme != "file":
        return input_uri

    path = parsed.path if parsed.scheme == "file" else input_uri
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return f"file://{path}"


def add_audio_source_branch(pipeline, index, uri, file_loop):
    source = f"source-{index}"
    queue = f"audio-queue-{index}"
    converter = f"audio-convert-{index}"
    resample = f"audio-resample-{index}"

    source_properties = {
        "uri": normalize_uri(uri),
        "disable-audio": False,
    }
    if file_loop:
        source_properties["file-loop"] = True

    pipeline.add("nvurisrcbin", source, source_properties)
    pipeline.add("queue", queue)
    pipeline.add("audioconvert", converter)
    pipeline.add("audioresample", resample)
    pipeline.link((source, queue), ("asrc_%u", ""))
    pipeline.link(queue, converter, resample)
    return resample


def build_pipeline(args):
    inputs = args.inputs or [AUDIO_FILE, AUDIO_FILE]
    batch_size = args.batch_size or len(inputs)
    if batch_size < len(inputs):
        raise ValueError("batch size must be greater than or equal to the number of inputs")

    pipeline = Pipeline(PIPELINE_NAME)
    pipeline.add("nvstreammux", "mux", {"batch-size": batch_size})

    for index, uri in enumerate(inputs):
        branch_tail = add_audio_source_branch(pipeline, index, uri, args.file_loop)
        pipeline.link((branch_tail, "mux"), ("", "sink_%u"))

    audio_caps = nvmm_audio_caps(args.audio_input_rate)
    pipeline.add("capsfilter", "infer-audio-caps", {"caps": audio_caps})
    pipeline.add("nvinferaudio", "audio-classifier", {
        "config-file-path": args.infer_config,
        "batch-size": batch_size,
        "audio-framesize": args.audio_framesize,
        "audio-hopsize": args.audio_hopsize,
        "audio-transform": args.audio_transform,
    })
    pipeline.add("fakesink", "sink", {"sync": False})

    pipeline.link(
        "mux",
        "infer-audio-caps",
        "audio-classifier",
        "sink"
    )

    if args.print_interval:
        pipeline.attach(
            "audio-classifier",
            Probe("audio-classification-printer", AudioClassificationPrinter(args.print_interval))
        )
    return pipeline


def run(args):
    if os.environ.get("USE_NEW_NVSTREAMMUX", "").lower() not in {"yes", "1", "true"}:
        raise RuntimeError("Infer mode requires nvstreammux2. Set USE_NEW_NVSTREAMMUX=yes in the environment.")

    build_pipeline(args).start().wait()


def parse_args():
    parser = argparse.ArgumentParser(description="PyServiceMaker Pipeline API audio inference sample")
    parser.add_argument("inputs", nargs="*", help="audio source URI or file path")
    parser.add_argument("--infer-config", default=SONYC_INFER_CONFIG, help="nvinferaudio config file")
    parser.add_argument("--batch-size", type=int, help="audio batch size; defaults to number of inputs")
    parser.add_argument("--audio-input-rate", type=int, default=SONYC_AUDIO_RATE, help="audio rate before nvinferaudio")
    parser.add_argument("--audio-framesize", type=int, default=SONYC_AUDIO_FRAME_SIZE, help="nvinferaudio frame size")
    parser.add_argument("--audio-hopsize", type=int, default=SONYC_AUDIO_HOP_SIZE, help="nvinferaudio hop size")
    parser.add_argument("--audio-transform", default=SONYC_AUDIO_TRANSFORM, help="nvinferaudio audio-transform structure")
    parser.add_argument("--file-loop", action="store_true", help="loop file inputs, matching the legacy deepstream-audio test config")
    parser.add_argument("--print-interval", type=int, default=1, help="print classifier labels every N inference batches; 0 disables printing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process = Process(target=run, args=(args,))
    try:
        process.start()
        process.join()
        if process.exitcode is None:
            sys.exit(1)
        if process.exitcode != 0:
            sys.exit(process.exitcode)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()
        process.join()
        sys.exit(130)
