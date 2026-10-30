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

# nvmodelmux test suite

Functional and stress tests for the DeepStream `nvmodelmux` plugin. Each case
drives a real `deepstream-test5-app` pipeline through the plugin's REST control
plane, then verifies the outcome from two sides: the REST responses, and the
plugin's own progress markers in the application log.

59 cases run by default (1–59), plus case 0 as a diagnostic, and a separate
suite of seven long-running soak workloads.

## Setup

Once per machine:

```bash
cd tests
./test_setup.sh
```

`test_setup.sh` checks that the plugin is installed and builds the five TensorRT
engines the cases need. That takes a few minutes the first time; afterwards the
engines are cached and reused. `run_tests.sh` runs this check itself on every
start, so you rarely need to invoke it directly.

## Running tests

### Base cases — run all

```bash
./run_tests.sh all      # or just: ./run_tests.sh
```

Cases 1–59, each in a fresh application process. Takes roughly 55 minutes — every
case starts a real pipeline and runs it. Note that case 0 is *not* part of `all`;
it is a diagnostic, run it explicitly.

### Base cases — run one

```bash
./run_tests.sh 5        # case number from the table below
./run_tests.sh 0        # the isolation probe
```

A single case leaves the application running at the end if the case itself does
not stop it, so you can poke at the pipeline by hand. Press Ctrl-C to finish.

### Base cases — two-terminal mode

To drive an application you started yourself:

```bash
# terminal 1
deepstream-test5-app -c ../configs/test5_config_multimodel.txt

# terminal 2
AUTO_APP=0 ./run_tests.sh 5
```

The harness then issues only the REST calls. Log-based assertions are skipped,
because the application log is not the harness's to scope.

### Stress workloads

Separate from the numbered cases, with their own entry point:

```bash
./run_stress.sh         # every workload (0-7), in order
./run_stress.sh 3       # one workload
./run_stress.sh 1       # its no-modelmux control
./run_stress.sh list    # show them, run nothing
```

`run_stress.sh` invokes `stress_setup.sh` itself on every run, so the stress
engines are built for you; you only need it directly to pay that cost up front.

Each workload holds the pipeline for `STRESS_DURATION` seconds (default 3600) and
samples memory, GPU, threads and file descriptors throughout, so the full set
takes hours. Shorten it for a smoke run:

```bash
STRESS_DURATION=300 ./run_stress.sh 3
```

The workloads are listed under [Stress workloads](#stress-workloads-1) below.

### Results

Every run prints a PASS/FAIL summary and exits non-zero if anything failed. The
per-case ledger is at `.logs/test_results.txt`.

## Base test cases

| # | What it verifies |
|---:|---|
| 1 | isolation probe: dynamic add/remove, inference disabled |
| 2 | custom primary requested WHILE warming -> passthru-with-tag, promote ALL waiting streams |
| 3 | drain-to-zero refill, single stream, repeated (lifecycle) **·loops** |
| 4 | drain-to-zero refill, full 4-stream batch **·loops** |
| 5 | per-model capacity -> auto-shard (grow) + auto-compact |
| 6 | metadata-driven primary routing alongside default |
| 7 | model load/unload lifecycle + serving guard |
| 8 | live scoped PRIMARY reroute (zero-drop) |
| 9 | default primary + default SHADOW baseline (+ overrides) |
| 10 | default p&s + custom p&s, mixed per stream |
| 11 | per-sensor routing via CONFIG [stream-model-*] |
| 12 | model/status GET (full + filtered) |
| 13 | live per-stream model/routing (reroute/reject/clear/rollback) |
| 14 | PROMOTE both-default (default rotates to promoted model) |
| 15 | PROMOTE non-default shadow over DEFAULT primary |
| 16 | PROMOTE custom shadow over CUSTOM primary (displaced custom unloaded) |
| 17 | PROMOTE DEFAULT shadow over CUSTOM primary (default_shadow cleared) |
| 18 | group-local PROMOTE (independent A/B groups) |
| 19 | SWAP both-default: defaults EXCHANGE (dp<->ds) |
| 20 | SWAP def_p/custom: custom ACQUIRES default_primary |
| 21 | SWAP custom/custom: NEITHER a default -> configured defaults UNTOUCHED |
| 22 | SWAP custom/def_s: custom ACQUIRES default_shadow |
| 23 | scoped SWAP: create+warm fallback, defaults UNTOUCHED |
| 24 | group-local SWAP (independent A/B groups) |
| 25 | dedup compaction ON (default): demote-to-limbo + cross-role reuse |
| 26 | dedup compaction OFF: keep role-tagged, no cross-role reuse |
| 27 | model-sharding=0: over-cap streams pass through (no inference) |
| 28 | optional default: no-model streams pass through, mixed with inference |
| 29 | perf metrics: attach-perf-metric -> fps + latency in model/status |
| 30 | buffer-copy-mode 0/1/2/3 -> per-lane nvvideoconvert copy |
| 31 | A/B swap with asymmetric branch rates — anti-stall (swap-display timeout) **·loops** |
| 32 | teardown WHILE reconfig in flight — UAF tripwire **·loops** |
| 33 | downstream source-id/pad-index restore after partial remove |
| 34 | duplicate add/remove + add-cancel-readd race tripwire **·loops** |
| 35 | multi-model config-bound custom primary lazy-load/reuse anti-stall **·loops** |
| 36 | rapid churn + repeated mid-churn teardown (lifecycle-race tripwire) **·loops** |
| 37 | reserved-separator model names rejected at register |
| 38 | perf observation: one 30-stream primary ModelBin |
| 39 | perf observation: 15-stream primary + 15-stream shadow A/B bins |
| 40 | model/update in-place OTA checkpoint transition (v1 -> v2 -> v3) |
| 41 | default switch sweeps ONLY default-routed streams; explicitly pinned streams stay — via_default/origin |
| 42 | per-model/per-version batch-size: sizing, auto-shard, immutability, clamp, preflight |
| 43 | pipeline blocks on the default engine build; defaults ready before streaming, no passthrough |
| 44 | model/routing GET: grouped routing snapshot (merge, origin split, filter) |
| 45 | model/routing GET: origin + default{} sweep (pinned lane is NOT swept) |
| 46 | model/routing GET: pending + passthrough states |
| 47 | model/routing GET: accepted-but-deferred route reports serving + pending together |
| 48 | model/routing: a newer deferred route supersedes an older one (warm order cannot decide the winner) |
| 49 | model/routing GET: a deferred shadow-clear is an explicit null, distinct from "leave unchanged" |
| 50 | model/routing: a route that applies immediately supersedes an older pending one (no later drag-back) |
| 51 | model/routing: superseding is per lane — a shadow-only route keeps the pending primary change |
| 52 | model/routing: a passthrough stream's newer queued route outranks its older promote tag |
| 53 | HTTP status + error-code contract: 404 vs 409 vs 400, error{} envelope, model{}+state on accepted ops, and plane separation (model/status reports the default designation, not the fleet routing summary) |
| 54 | unload a NON-DEFAULT version while a sibling version of the same name is the configured default: refused for SERVING (not the name-level default rule), succeeds once idle, default sibling untouched and still protected |
| 55 | default pair transitions from primary=v1/shadow=v2: SWAP (roles exchange), PROMOTE (`"shadow": null`, the A/B ends), and the both-roles CONFLICT refused with 409 + MODEL_IN_USE before anything mutates. Both models are already warm, so each transition must RELABEL the existing bins rather than build new ones — asserted by bin identity (`gie=`) per stream per role, not just by the absence of a reload |
| 56 | STATIC config-listed streams (`list=` + `sensor-id-list` + `sensor-name-list`): a stream created from the app config must end up with the same identity a `stream/add` stream gets — its camera_id resolves in both GET planes and in per-stream filters, and no stream falls back to the unnamed `src` placeholder |
| 57 | STATIC streams with NO config default: the other bind/attach ordering — identity must survive a bind that arrives after the stream is already wired |
| 58 | model/load artifact rule: the FIRST version of a name may load bare (the catalog base config is its artifact), a SECOND version with no `engine_file`/`config_file`/`[model-<name>-<version>]` is refused at admission with `ARTIFACT_MISSING` — otherwise two version numbers resolve to one model |
| 59 | SECONDARY stage control plane (`[secondary-gie0]`, `plugin-type=4`, `process-mode=2`): `model/update` OTA and `model/load` are accepted and confirmed at this position, while the machinery it does NOT have is refused by name rather than generically — per-stream routing (`ROUTING_PER_STREAM_UNSUPPORTED`), a shadow reference (`ROUTING_SHADOW_UNSUPPORTED`), a route to a non-resident version, and unload of the only serving model. An all-streams route promotes the idle resident version and the displaced one stays warm, so the reversal is instant. `model/status` and `model/routing` report the stage that answered without leaking a secondary entry into `routes[]`. A rejected route leaves `routing_revision` untouched; an applied one advances the counter *shared* by every stage. Finally a route to a still-warming rebuild is PARKED: the previous model keeps serving, unload of the parked target is refused naming the route rather than the build, and the route applies once the build finishes |

The **·loops** cases repeat their cycle `RACE_REPS` times (default 5) to catch
faults that only appear intermittently.

Case 59 is the only case that drives a **secondary** stage. It enables
`[secondary-gie0]` in the application config as a second `nvmodelmux` serving a
vehicle-type classifier on the objects the primary detected. Its model, engine and
nvinfer config come from `SGIE_MODEL` / `SGIE_ENGINE` / `SGIE_INFER_CONFIG`. The
catalog entry points at the nvinfer *config* rather than a prebuilt engine, so loading
a new version rebuilds from ONNX — that is what opens a real warming window for the
parked-route assertions at the end of the case.

## Stress workloads

Run with `./run_stress.sh <n>`.

The workloads come in **matched pairs**. `[primary-gie]` says which element is
actually in the pipeline: tests 0 and 1 run **plain nvinfer**, tests 2-7 run
**nvmodelmux**. A control and its twin are otherwise identical — same cycles,
holds, batch, source and inference load — so the only difference between them is
which element sits in that slot.

That is what makes them diagnostic: if a control shows the same growth or the
same failure as its twin, it is not modelmux; it is DeepStream's own decode /
streammux / source-churn path.

| # | `[primary-gie]` | Workload | What it looks for |
|---:|:---|---|---|
| 0 | **nvinfer** | drain-to-zero add/remove churn, `STRESS_CYCLES` cycles | control for 2 — is per-cycle teardown accumulation modelmux's, or DeepStream's? |
| 1 | **nvinfer** | soak + async random churn, `NUM_STREAMS` streams | control for 3 — same question against a pipeline that never drains |
| 2 | **nvmodelmux** | drain-to-zero add/remove churn, `STRESS_CYCLES` cycles | anything teardown fails to release, accumulating once per cycle. Empties to **zero** streams and refills, so it exercises the inner mux/demux recreate path the soak tests never reach |
| 3 | **nvmodelmux** | soak + async random churn, `NUM_STREAMS` streams | the main run — churn against a live pipeline that never drains |
| 4 | **nvmodelmux** | A/B steady soak (worst-int8 primary / best-fp32 shadow), no churn | whether two live inference lanes accumulate while sitting still |
| 5 | **nvmodelmux** | A/B soak + async churn | attach/detach cost of the second lane (read against 4) |
| 6 | **nvmodelmux** | two primaries steady, no swap | the cost of hosting two models (baseline for 7) |
| 7 | **nvmodelmux** | two primaries with live model-swap | the cost of moving streams between them, isolated from add/remove |

`nvinfer` = `plugin-type=0` (plain inference, no model plane).
`nvmodelmux` = `plugin-type=4` (routing, A/B lanes, live model swap).

Read them in pairs — that is where the diagnosis comes from:

- **2 against 0** and **3 against 1** separate plugin growth from DeepStream's own
  churn cost. These are the attribution pairs: a failure that reproduces in the
  control is not a modelmux bug;
- **5 against 4** separates lane attach/detach from steady-state inference;
- **7 against 6** separates rerouting from merely hosting two models.

`PGIE_ENABLE` applies to tests **0 and 1 only** — it decides whether their
nvinfer actually runs:

- `1` (default) — nvinfer runs, so the inference load matches the modelmux twin
  and the pair differs in exactly one thing: nvinfer vs nvmodelmux.
- `0` — `[primary-gie]` is disabled entirely, giving the bare
  `nvmultiurisrcbin -> streammux -> tiler -> sink` pipeline with no inference at
  all. Useful *after* something reproduces, to find out whether inference was
  needed to trigger it.

Default `1` on purpose: with `0` a control differs from its twin in **two** ways
(no modelmux *and* no inference), so a clean run cannot tell you which of them
mattered.

Results:

```
.logs/stress_results.txt   one line per workload
.logs/stress/*.csv         the resource samples
.logs/stress_<n>_logs.txt  the full run log
```

## Results and logs

Everything for one case lands in a single file:

```
.logs/test_<n>_logs.txt
```

It holds the harness narration — config edits, REST request and response,
`[PASS]`/`[FAIL]` lines — interleaved with the application's own stdout and
stderr, in chronological order. A REST call and the plugin's reaction to it sit
next to each other, which is usually enough to diagnose a failure without
reproducing it.

```
.logs/test_results.txt   PASS/FAIL ledger, one block per case
.logs/*.txt              the throwaway config copies used for the run
```

The whole `.logs/` tree is git-ignored.

## Scripts

| Script | Purpose |
|---|---|
| `run_tests.sh` | the functional cases — this is the entry point |
| `test_setup.sh` | preflight: plugin installed, TensorRT engines built |
| `frameacct_check.sh` | the post-case sanity gates, sourced by `run_tests.sh` |
| `run_stress.sh` | long-running stress workloads |
| `stress_setup.sh` | engine preflight for the stress workloads |
| `mem_monitor.sh` | memory/leak sampling, used by `run_stress.sh` |

## How a case works

Two configs are in play, and the checked-in originals are never touched:

- **main** — the test5 config (`[source-list]`, `[streammux]`, `[primary-gie]` with
  `plugin-type=4`, `[tiled-display]`, `[sink0]`);
- **mm** — the plugin's own `[multimodel]` config.

Each case copies both into the run directory, edits only the copies (`cfg_set`
routes a group to the right file), starts the application against them, drives
REST, and asserts on the responses and on the plugin's markers in the log.

Every case then passes through `run_default_sanity_checks`: zero frame loss, no
inference stall, no GStreamer-CRITICAL/assert/crash, no frame gaps, and source-id
restored on every output frame. These are the checks no single case asserts,
because they must hold for all of them.

## Environment overrides

```bash
HOST=localhost PORT=9000 DELAY=5 ./run_tests.sh 5
AUTO_APP=0 ./run_tests.sh 5           # drive an application started elsewhere
RACE_REPS=2 ./run_tests.sh 3          # shorter stress cycles
ENABLE_DISPLAY=1 ./run_tests.sh 5     # EGL sink instead of fakesink
RUN_TEST_SETUP=0 ./run_tests.sh all   # skip the engine precheck
APP_BIN=/path/to/deepstream-test5-app ./run_tests.sh 1
MM_PERF_DWELL_SEC=20 ./run_tests.sh 38   # shorter perf dwell
SGIE_MODEL=VehicleTypes ./run_tests.sh 59  # secondary-stage model name
```

## Concurrency

The suite takes an exclusive lock (`.logs/.run_tests.lock`) for the whole run. Two
runs would share the REST port, the run directory and the application binary, and
would corrupt each other's results rather than fail visibly — so the second one
aborts with a message instead. The lock releases when the shell exits, however it
exits.

## Troubleshooting

**`another nvmodelmux test run is already running`.** A run still holds the lock,
or one was killed leaving its application behind. Check for a stray
`deepstream-test5-app` and stop it.

**Several cases fail at once and it is not clear which layer is at fault.** Run
case 0 first:

```bash
./run_tests.sh 0
```

It adds, drains and re-adds streams with inference switched off entirely, so it
exercises only the stream lifecycle. If it fails, the fault is below the plugin
and the model-plane cases are failing for that reason rather than their own. If it
passes, the lifecycle is sound and the failures are genuinely about routing or
models.
