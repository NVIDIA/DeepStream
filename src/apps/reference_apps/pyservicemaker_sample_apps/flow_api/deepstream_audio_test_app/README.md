<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

This Flow API sample demonstrates the PyServiceMaker version of the legacy
`deepstream-audio` SONYC classifier app on DeepStream `rel-39`.

By default it follows the same audio inference shape as
`apps/deepstream/sample_apps/deepstream-audio`:

```text
audio URI sources -> nvstreammux -> nvinferaudio -> fakesink
```

The sample uses `Flow.capture(..., media_types=MediaType.AUDIO)` to expose
`nvurisrcbin.asrc_%u`, batches the audio streams with `Flow.batch()`, and runs
the SONYC model through `Flow.infer()` / `nvinferaudio`. It also keeps the
simpler audio helper modes for playback, RTSP output, and WAV encoding.

Set `USE_NEW_NVSTREAMMUX=yes` before running the infer mode because audio
batching requires nvstreammux2. The sample expects DeepStream SDK assets under
the standard installed SDK tree at `/opt/nvidia/deepstream/deepstream`.

Examples:

```bash
# Run the SONYC audio classifier using the same WAV twice, matching the legacy
# deepstream-audio test configuration's two-source batch.
USE_NEW_NVSTREAMMUX=yes python3 deepstream_audio_test.py

# Run inference on explicit inputs.
USE_NEW_NVSTREAMMUX=yes python3 deepstream_audio_test.py \
  /opt/nvidia/deepstream/deepstream/samples/streams/sonyc_mixed_audio.wav \
  /opt/nvidia/deepstream/deepstream/samples/streams/sonyc_mixed_audio.wav

# Loop file inputs, similar to the legacy [tests] file-loop=1 setting.
USE_NEW_NVSTREAMMUX=yes python3 deepstream_audio_test.py --file-loop

# Exercise the non-infer audio helper modes.
python3 deepstream_audio_test.py --mode discard
python3 deepstream_audio_test.py --mode play
python3 deepstream_audio_test.py --mode stream --rtsp-port 8556 --rtsp-mount-point ds-audio
python3 deepstream_audio_test.py --mode encode --output ~/sonyc_copy.wav
```

The default inference settings mirror
`apps/deepstream/sample_apps/deepstream-audio/configs/ds_audio_sonyc_test_config.txt`:

- `config_infer_audio_sonyc.txt`
- `batch-size=2` when using the default inputs
- `audio-transform=melsdb,...,sample_rate=44100,...`
- `audio-framesize=441000`
- `audio-hopsize=110250`

For mixed audio/video capture, call `select(MediaType.AUDIO)` or
`select(MediaType.VIDEO)` on the returned `Flow` before attaching downstream
helpers. If the source already
contains both pads and the goal is one muxed container output, the Flow API now
also supports:

```bash
python3 -c 'from pathlib import Path; from pyservicemaker import Flow, MediaType, Pipeline; Flow(Pipeline("mux")).capture(["/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"], media_types=(MediaType.VIDEO, MediaType.AUDIO)).encode(str(Path.home() / "sample_720p_muxed.mp4"))()'
```
