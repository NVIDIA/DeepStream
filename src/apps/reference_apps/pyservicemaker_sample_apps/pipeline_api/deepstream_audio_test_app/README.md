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

This sample demonstrates the PyServiceMaker Pipeline API version of the legacy
`deepstream-audio` SONYC classifier app.

Unlike the Flow API sample, this app builds the audio graph explicitly with
`Pipeline.add()` and `Pipeline.link()`:

```text
nvurisrcbin.asrc_%u
  -> queue
  -> audioconvert
  -> audioresample
  -> nvstreammux.sink_%u
  -> capsfilter(audio/x-raw(memory:NVMM), rate=44100, format=S16LE, layout=interleaved, channels=1)
  -> nvinferaudio
  -> fakesink
```

Audio batching requires nvstreammux2, so set `USE_NEW_NVSTREAMMUX=yes` before
running this sample. The sample expects DeepStream SDK assets under the standard
installed SDK tree at `/opt/nvidia/deepstream/deepstream`.

The default settings mirror
`apps/deepstream/sample_apps/deepstream-audio/configs/ds_audio_sonyc_test_config.txt`:

- two default `sonyc_mixed_audio.wav` sources
- `batch-size=2`
- `config_infer_audio_sonyc.txt`
- `audio-transform=melsdb,...,sample_rate=44100,...`
- `audio-framesize=441000`
- `audio-hopsize=110250`

Examples:

```bash
# Run the SONYC classifier using the same WAV twice.
USE_NEW_NVSTREAMMUX=yes python3 deepstream_audio_test.py

# Run on explicit audio inputs.
USE_NEW_NVSTREAMMUX=yes python3 deepstream_audio_test.py \
  /opt/nvidia/deepstream/deepstream/samples/streams/sonyc_mixed_audio.wav \
  /opt/nvidia/deepstream/deepstream/samples/streams/sonyc_mixed_audio.wav

# Loop file inputs, similar to the legacy [tests] file-loop=1 setting.
USE_NEW_NVSTREAMMUX=yes python3 deepstream_audio_test.py --file-loop

# Disable label printing and just validate graph/runtime execution.
USE_NEW_NVSTREAMMUX=yes python3 deepstream_audio_test.py --print-interval 0
```

This sample is intended as the lower-level escape hatch counterpart to
`../../flow_api/deepstream_audio_test_app`.
