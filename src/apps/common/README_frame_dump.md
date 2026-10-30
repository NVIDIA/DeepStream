<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frame Dump — Runtime JPEG + Metadata Dump

A DeepStream feature that saves **JPEG frames + per-object detection/tracker
metadata** from a running pipeline. It can be driven two ways — from the app
**config file** (`[frame-dump]` group) at startup, or over a small **REST API**
at runtime. Both expose the exact same controls.

- **Where it hooks:** a probe on the **nvtracker** src pad, so a tracker must be
  enabled in the pipeline (it needs the detection + tracker metadata).
- **Shared wiring:** enabled once in the common `deepstream_app.c`
  (`create_common_elements`), so `deepstream-app`, `deepstream-test5`, and
  `metropolis_perception_app` all get it with no per-app code.
- **Module:** `apps/deepstream/common/{includes,src}/frame_dump.{h,cpp}`.

## Output layout
```
<location>/<sensor_id>/<YYYY-MM-DD>[/<HH>]/<frame_num>.jpg
<location>/<sensor_id>/<YYYY-MM-DD>[/<HH>]/<frame_num>.txt
```
- `.jpg` — the full frame, GPU-encoded (quality configurable).
- `.txt` — JSON: frame info (`sensor_id`, `source_id`, `frame_num`, `buf_pts`,
  `ntp_timestamp`, `wall_time`, `frame_width/height`) plus, per object,
  `class_id`, `label`, `object_id` (tracker id), `confidence`,
  `detector_bbox`, `tracker_bbox`.
- `<sensor_id>` comes from the source config (`sensor-id-list` for
  nvmultiurisrcbin, or `sensor-id` under `[sourceN]`). If unset it falls back to
  `source_<N>`.

---

## 1. Static config — the `[frame-dump]` group

Add the group to the app config passed with `-c`. The DeepStream parser ignores
this unknown group, and the frame-dump module reads it (works for both `.txt`
GKeyFile and YAML configs). **All keys are optional**; the feature starts
**disabled** unless `enable=1`.

### `.txt` (GKeyFile) form
```ini
[frame-dump]
enable=1                          # turn the feature on
fps=1                             # dumps per second of *video* time, per source
location=/tmp/frame_dumps         # output root directory
all-cameras=1                     # dump every source (ignores per-camera list)
quality=80                        # JPEG quality 1-100
hour-subfolder=0                  # add an /<HH>/ level under the day folder
unique-filenames=0                # name files <frame_num>_<pts> (survive restarts)
retention-days=0                  # delete day-folders older than N days; 0 = keep
enable-sensors=UniqueSensorId1;UniqueSensorId2   # cameras ON at startup (";"-list)
port=9857                         # REST control-server port
bind-address=0.0.0.0              # REST bind ("127.0.0.1" for local-only)
kafka-enable=0                    # publish a Kafka message per dumped frame
kafka-broker=localhost;9092       # "host;port" (startup only)
kafka-topic=frame-dumps           # notification topic (different from any other)
kafka-image=0                     # 1 = no files; send base64 JPEG over kafka
kafka-image-topic=frame-images    # topic for base64-image msgs (empty->kafka-topic)
#kafka-proto-lib=...              # adaptor .so; default libnvds_kafka_proto.so
#kafka-config-file=...            # optional Kafka producer config file
```

### YAML form
```yaml
frame-dump:
  enable: 1                       # 1/0 or true/false both accepted
  fps: 1
  location: /tmp/frame_dumps
  all-cameras: 1
  quality: 80
  hour-subfolder: 0
  unique-filenames: 0
  retention-days: 0
  enable-sensors: UniqueSensorId1;UniqueSensorId2
  port: 9857
  bind-address: 0.0.0.0
  kafka-enable: 0
  kafka-broker: localhost;9092
  kafka-topic: frame-dumps
  kafka-image: 0
  kafka-image-topic: frame-images
```

### Key reference
| Config key | REST field | Meaning | Default |
|---|---|---|---|
| `enable` | `enable` | master on/off | 0 |
| `fps` | `fps` | dumps per second of video (PTS) time, per source; fractional values below 1 give sparse sampling (e.g. `0.1` = one frame every 10s) | 1 |
| `location` | `location` | output root | `/tmp/frame_dumps` |
| `all-cameras` | `all_cameras` | dump every source | 0 |
| `enable-sensors` | `sensors` (array) | specific cameras to dump | (none) |
| `quality` | `quality` | JPEG quality 1-100 | 80 |
| `hour-subfolder` | `hour_subfolder` | add `/<HH>/` level | 0 |
| `unique-filenames` | `unique_filenames` | `<frame_num>_<pts>` names | 0 |
| `retention-days` | `retention_days` | auto-delete old day folders (0=keep) | 0 |
| `port` | — (startup only) | REST server port | 9857 |
| `bind-address` | — (startup only) | REST bind address | 0.0.0.0 |
| `kafka-enable` | `kafka_enable` | publish a Kafka message per dumped frame | 0 |
| `kafka-broker` | — (startup only) | Kafka `"host;port"` connection string | (none) |
| `kafka-topic` | `kafka_topic` | notification topic | (none) |
| `kafka-image` | `kafka_image` | send base64 JPEG over kafka; **no `.jpg`/`.txt` files** | 0 |
| `kafka-image-topic` | `kafka_image_topic` | topic for base64-image msgs (empty → `kafka-topic`) | (none) |
| `kafka-proto-lib` | — (startup only) | Kafka adaptor `.so` | `libnvds_kafka_proto.so` |
| `kafka-config-file` | — (startup only) | Kafka producer config file | (none) |

> Config values only **seed** the state at startup. Everything except `port` /
> `bind-address` / `kafka-broker` / `kafka-proto-lib` / `kafka-config-file` can
> still be changed live over REST.

## Kafka notification
When `kafka-enable=1` (and `kafka-broker` is set), every dumped frame publishes
one JSON message to `kafka-topic` — a **different topic** from any inference
message broker — informing consumers which files were written:
```json
{
  "event": "frame_dumped",
  "sensor_id": "UniqueSensorId1",
  "source_id": 0,
  "frame_num": 1292,
  "buf_pts": 43133333333,
  "ntp_timestamp": 0,
  "wall_time": "2026-07-28T09:15:04",
  "jpg_path": "/data/frame_dumps/UniqueSensorId1/2026-07-28/1292.jpg",
  "txt_path": "/data/frame_dumps/UniqueSensorId1/2026-07-28/1292.txt",
  "num_objects": 7
}
```
It uses the DeepStream **nvds_msgapi Kafka protocol adaptor**, loaded with
`dlopen()` at runtime — no extra link dependency. The message is sent from the
async writer thread, so it never stalls the streaming path. Toggle it live:
```bash
curl -X POST http://localhost:9857/dump/config \
  -d '{"kafka_enable":true,"kafka_topic":"frame-dumps"}'
```
(The broker is fixed at startup; only `kafka_enable` and `kafka_topic` are
runtime-changeable.)

## Kafka base64-image mode
When `kafka-image=1` (and `kafka-broker` is set), the feature **stops writing
`.jpg`/`.txt` files** entirely. Instead, for every frame that passes the same
fps / per-camera selection, it JPEG-encodes the frame **in memory** (GPU
encoder, no disk), base64-encodes it, and publishes one JSON message —
frame metadata **plus** the image — to `kafka-image-topic` (a **separate topic**;
falls back to `kafka-topic` if unset). The payload is the same metadata schema
as the `.txt` with an added `image` field:
```json
{
  "sensor_id": "UniqueSensorId1",
  "source_id": 0,
  "frame_num": 1292,
  "buf_pts": 43133333333,
  "ntp_timestamp": 0,
  "wall_time": "2026-07-28T09:15:04",
  "frame_width": 1920,
  "frame_height": 1080,
  "objects": [ { "class_id": 0, "label": "person", "object_id": 12,
                 "confidence": 0.94, "detector_bbox": {...}, "tracker_bbox": {...} } ],
  "image": "<base64-encoded JPEG bytes>"
}
```
The in-memory JPEG comes back attached to the frame as `NVDS_CROP_IMAGE_META`
after `nvds_obj_enc_finish()`. It is base64-encoded inline in the probe (the
encoded buffer lives on the batch meta and is freed once the buffer flows
downstream), then the resulting message is handed to the async writer thread to
publish, so the network send never stalls the streaming path. Toggle it live (no restart):
```bash
curl -X POST http://localhost:9857/dump/config \
  -d '{"enable":true,"kafka_image":true,"kafka_image_topic":"frame-images"}'
```
> **Message size:** a base64 1080p JPEG is ~130–400 KB depending on `quality`.
> If frames approach the broker limit, lower `quality` or raise the producer's
> `message.max.bytes` / `max.request.size` via `kafka-config-file`.
> `kafka-image` and `kafka-enable` are independent; in image mode the file-write
> and the file-path notification are both skipped (the probe takes the in-memory
> branch), so leave `kafka-enable=0` when using `kafka-image=1`.

---

## 2. REST Command API

The control server starts when the pipeline's tracker is created and listens on
`port` (default **9857**). Use `localhost` from inside the container (or if the
container is `--net=host`), otherwise the host IP.

### `GET /dump/status`
Returns the current settings and the list of cameras switched on.
```bash
curl http://localhost:9857/dump/status
```
```json
{ "enabled": true, "fps": 1, "quality": 80, "all_cameras": true,
  "hour_subfolder": false, "unique_filenames": false, "retention_days": 0,
  "location": "/tmp/frame_dumps", "cameras_on": [] }
```

### `POST /dump/config`
Set any subset of the global controls. Fields: `enable`, `fps`, `location`,
`quality`, `all_cameras`, `sensors` (array), `hour_subfolder`,
`unique_filenames`, `retention_days`, `kafka_enable`, `kafka_topic`,
`kafka_image`, `kafka_image_topic`.
```bash
# one call: turn on + start dumping EVERY stream
curl -X POST http://localhost:9857/dump/config \
  -d '{"enable":true,"fps":1,"location":"/data/frame_dumps","all_cameras":true}'

# one call: start dumping only specific cameras
curl -X POST http://localhost:9857/dump/config \
  -d '{"enable":true,"fps":1,"location":"/data/frame_dumps","sensors":["UniqueSensorId1"]}'

# change settings live
curl -X POST http://localhost:9857/dump/config -d '{"fps":5}'
curl -X POST http://localhost:9857/dump/config -d '{"quality":95}'
curl -X POST http://localhost:9857/dump/config -d '{"hour_subfolder":true}'

# stop dumping
curl -X POST http://localhost:9857/dump/config -d '{"enable":false}'
```

### `POST /dump/camera`
Toggle a single camera on/off (fine-grained; used when `all_cameras` is false).
```bash
curl -X POST http://localhost:9857/dump/camera -d '{"sensor_id":"UniqueSensorId1","enable":true}'
curl -X POST http://localhost:9857/dump/camera -d '{"sensor_id":"UniqueSensorId2","enable":false}'
```

---

## Notes

- **`fps` is video-time, not wall-clock.** The rate is gated on the mux
  `buf_pts`, so `fps=1` = one frame per second of the stream's own timeline. For
  live cameras that equals one per real second; for a fast-decoding file source
  it dumps that fast (still 1 per video second).
- **Sparse sampling (< 1 fps):** `fps` accepts fractional values, so
  `fps=0.1` dumps roughly one frame every 10 seconds and `fps=0.0167` roughly
  one per minute. Values `<= 0` fall back to the default of `1`.
- **Which cameras dump:** `all_cameras=true` dumps all; otherwise only cameras
  in the ON set (`sensors`/`enable-sensors`/`/dump/camera`).
- **A tracker is required** — the probe is on the tracker src pad. No tracker,
  no dumps (and the REST server won't come up).
- **Disk / retention:** at high fps use `hour-subfolder=1`; over multi-day runs
  set `retention-days=N` to auto-prune, and `unique-filenames=1` if sources can
  restart (frame numbers reset).

## Build
The module is compiled from `apps/deepstream/common/src/frame_dump.cpp` and needs
these libraries (already added to the app Makefiles):
```
-lnvds_batch_jpegenc -lpthread -lnvbufsurface   (json-glib via pkg-config json-glib-1.0)
```
It lives in the mixed C/C++ `apps-common` static lib, which builds with C++
exceptions/RTTI disabled, so the REST layer uses json-glib (C API) and the YAML
`frame-dump:` block is read with a small built-in scalar parser (no yaml-cpp).
No changes are needed in any app's `main()` — the feature is wired in the shared
`create_common_elements()`.
