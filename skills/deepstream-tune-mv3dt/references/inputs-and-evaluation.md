# Inputs and evaluation

## Intake

Accept recordings or authorized captures, existing DeepStream configs, detector assets/caches, calibration in a documented format, optional GT and an optional floorplan. Inventory what exists before asking questions. Camera calibration is required for MV3DT. Route missing or invalid calibration to the AMC workflow. Do not infer metric scale from an image or tune calibration against held-out tracking labels.

Create a prepared dataset in the campaign workspace. Give each camera a stable ID and map it explicitly to video, calibration, source stream and output file. Do not infer correspondences from sorted filenames unless verified. Preserve original files. Record class mapping, including detector class versus tracker model class. Person is not universally class 0 and vehicles need their own dimensions.

Verify horizontal and vertical units independently against authoritative dataset metadata and camera matrices. A metric-looking object height does not prove XY is in meters: a legacy benchmark can use scaled XY with metric Z. Then an XY matching tolerance and a mixed-coordinate camera-distance cutoff do not have their apparent physical meaning. Preserve and label legacy reproductions separately, derive any metric conversion from source definitions rather than fitting GT, and freeze a new protocol before physical-coordinate testing.

Probe codecs, dimensions, duration, frame count, average and nominal FPS, time base and PTS. Different average/nominal FPS is a reason to inspect PTS, not proof of variable frame rate. Check drops and duplicated frames. Preserve timing when making MP4s from image sequences. Record any dewarping, rotation, crop, resizing or padding and update calibration for that exact image transform.

Verify calibration pixel coordinates against the tracker's working resolution, not only the source video. In the reviewed path, the plugin scales detections into tracker pixels and the projector uses its supplied projection matrix directly. When the original calibration image space is known, a resize from width W/height H to W'/H' uses S = diag(W'/W, H'/H, 1), P' = S P and K' = S K, leaving world geometry unchanged. Do not rescale a matrix whose image-space provenance is unknown. Prefer a supported crop backend that keeps verified geometry unchanged.

Prepare synchronized clips covering the intended common interval. The budget uses physical recording time, not accelerated playback time. When `physical_fps` is supplied, inspection counts decoded presented frames and divides by that verified cadence. Packet/header counts can include invisible preroll after a stream-copy trim. Otherwise it uses video-stream duration or decoded video timestamps, never the container duration or an unverified duration tag, which can include longer audio or an absolute end time. Unknown timing requires a prepared timestamped copy or verified capture cadence.

Resolve reported timing warnings before asking for approval. Variable-rate recordings need explicit timestamp inspection or a documented normalized time axis. A physical_fps override is not proof of constant capture cadence. Equal durations also do not prove synchronization, so check capture clocks and visual events even when encoded start times match. For mixed durations, document the common interval and trim copies first. The harness refuses an inconsistent selected duration. For multiple sequences, use separate manifests and sum camera-minutes when requesting one overall budget. Allocate child deadlines under the approved parent deadline, not fresh full budgets.

## Canonical observations

The helper CSV schema is:

```text
camera,frame,id,class,x,y,left,top,width,height
```

- `camera`: exact manifest ID, treated as text.
- `frame`: one-based frame in the prepared video time axis, after an explicit mapping.
- `id`: lossless text identity. Preserve 64-bit IDs and valid ID 0. Different sequences have independent identity namespaces.
- `class`: normalized class ID, matching the manifest and calibration model.
- `x,y`: physical ground-plane positions in declared world units. Leave blank when absent, never fabricate them from unknown calibration.
- `left,top,width,height`: image pixels on the prepared video, optional for world evaluation and needed for box overlays.

Use `evaluate.py convert` for the sample nine-field GT format and native per-camera trajectory files. Convert each stream separately and combine the output CSVs with one header. Native trajectory files are not the generic KITTI format. Check columns against the writer in the selected tracker revision before using this adapter. JSON, MOTChallenge, CVAT, COCO, timestamped logs and other formats need a small session-local adapter with explicit field mapping. Preserve identity and class semantics, ignored regions and visibility conventions. Test several known frames and cameras visually, including frame 0/1 and the last frame.

Native trajectory class IDs come from field 10 (zero-based). For that format, `--class-id` filters observations rather than relabeling them. Omit it to preserve all native classes. For unlabelled nine-field GT, supply the known class explicitly. The current native writer masks target IDs with `0xffff`. The adapter preserves the emitted ID but cannot recover discarded bits. Verify that masked IDs remain unique in the recording, or instrument a lossless export before claiming global identity metrics. Do not merge ambiguous IDs by guesswork.

The application's KITTI output is a different writer and can contain both online 3D rows and past-frame rows with different fields. Inspect its source and select the declared output semantics explicitly. Preserve excluded rows separately for audit. Do not parse all rows with one guessed schema or mix past-frame exports into an online comparison.

Provide `--frame-map` when annotation indices differ from video indices. Its JSON is `{ "camera_id": { "original_frame": prepared_video_frame } }`. Every source annotation frame must be mapped. For WILDTRACK, annotation steps 1..40 map to prepared frames 1,6,...,196. Do not infer annotation time from the clips' encoded 25 FPS. SCOUT's prepared test clips already contain the intended last half, so do not halve them again.

## Evaluation contract

Declare the exact annotated frame set, including empty annotated frames, in `evaluation.frames`. Use a list or `{ "start": 1, "stop": 100, "step": 1 }`, with inclusive stop. Prediction rows outside this set are not scored. Missing predictions on included frames remain misses. Confirm the pipeline processed the full intended sequence separately from annotation coverage.

The reusable metric core is `src/apps/reference_apps/deepstream-tracker-3d-multi-view/utils/evaluate_tracking.py`. It calls pinned public TrackEval HOTA, CLEAR and Identity. Use the sample's scripts/setup_evaluation.sh, which installs its requirements.txt plus evaluation dependencies and the pinned TrackEval revision. Record actual package versions because the other dependencies are not all pinned. The skill wrapper reuses the metric formulas and independently supplies manifest-driven data instead of inheriting SCOUT/WILDTRACK camera counts, GT hashes, ROI or a 30-unit cutoff.

Set `world_units`, `class_id`, fusion (`closest`, `median` or `mean`), `zero_distance`, optional region and optional camera-distance cutoff explicitly. `null` means no camera-distance filter. Similarity is `max(0, 1 - distance / zero_distance)`, with CLEAR/Identity threshold 0.5. Thus zero_distance 2 gives a 1-unit matching tolerance. HOTA uses multiple thresholds. The camera cutoff is a different concept, applied symmetrically before aggregation. Count both eligible and admitted observations so filtering is visible.

Declare `evaluation.output_semantics`, such as online frame metadata or native retrospective trajectories with named state/visibility filters. It enters the protocol signature. Reports without this declaration can be inspected but not automatically ranked. The campaign summary also separates different job hashes or launch scopes, preventing accidental ranking across live/replay or diagnostic/production launch changes. Report deliberately matched cross-scope comparisons separately with their evidence.

Use closest fusion as the default proposal for a new world-space protocol, then freeze it. Honor an established benchmark's required fusion instead. Do not tune fusion, ROI, distance cutoff or matching tolerance alongside tracker parameters and call the resulting difference tracker improvement. Rendering may use a different fusion, clearly labeled and not fed into evaluation.

Score each object class separately. Do not average unrelated sequence/class HOTA percentages as an official aggregate. For multi-sequence aggregate scores, use TrackEval's metric-specific sequence combination after validating the adapter. The bundled wrapper scores one class and one prepared sequence per invocation.

If GT has only 2D boxes, use the pinned TrackEval MOTChallenge adapter for per-camera IoU evaluation. Preserve its matching and ignore-region convention. Without cross-camera identities or physical positions, do not claim global IDF1 or world-space accuracy. This requires a dataset-specific adapter, not invented world coordinates. Document the narrower evaluation scope.

For that image-space path, remap GT and prediction IDs separately to dense per-camera integers before the legacy adapter casts whole rows to float or allocates by maximum ID. Save reversible maps from the original text IDs, including valid ID 0 and uint64 values. Map sparse annotation times explicitly to dense evaluation steps, retaining empty annotated frames and fixed sequence lengths. Use TrackEval's sequence combination rather than arithmetic camera averages.

The pinned MOTChallenge adapter still contains np.float/np.int aliases that newer NumPy versions removed. The sample setup only patches the world-metric modules it needs. Reproduce any compatibility failure, then use a documented process-local compatibility adapter or isolated patched copy, recording its source hash. Do not silently modify the installed environment. Keep explicit result columns because combined Count results omit the per-sequence Frames field.

Separate tuning and held-out sequences where available. With only one recording, use contiguous temporal splits with warm-up context, not random adjacent frames. Report test-set tuning when no holdout can be created. Do not claim a matched published SOTA comparison from the packaged sample protocols.

Inspect GT eligibility against raw video before interpreting false positives. Visible but unlabeled people can be counted as FP under a supplied-label protocol. Confirm the dataset's authoritative annotation area and ignore policy, do not infer an ROI from current GT extents or tune thresholds to suppress real unlabeled subjects. If coverage is unresolved, qualify the provided-label scores and request the missing protocol facts. Preserve earlier results when evaluating a separately justified protocol.

## Runtime and output semantics

Record source FPS, processed tracker FPS, detector interval, annotation cadence and timestamp source separately. Frame-based parameters often advance on tracker updates, including prediction-only frames, not just detection frames. Inspect the relevant counter before converting seconds into frames. Pose interval N typically means every N+1 eligible target ages, with tentative/ID initialization exceptions. Inspect ReID extraction cadence independently.

Freeze transport and message synchronization between candidates. Communicator.communicatorType 1 is the in-memory debug communicator for streams within the same low-level tracker instance, not across separate instances or sub-batches. Type 2 uses MQTT. Before enabling MultiViewAssociator.enableMsgSync, verify equal processed frame counts and coordinated end-of-stream across every peer video. A peer that ends early stops sending SYNC messages and can stall the remaining cameras. Equal durations alone are insufficient, and this setting does not replace capture-clock alignment.

Freeze online versus saved-trajectory evaluation. Native trajectory dumps can backfill tentative tracks and retrospectively update identities. Dump filters and online output filters are not interchangeable. Count and investigate same-camera duplicate frame/ID observations, especially after late re-association, before adapting a format.

The generic scorer rejects ambiguous duplicates. A benchmark's released scorer may admit them with a specific aggregation rule. Reproducing that rule is a separately documented adapter/protocol decision, not permission to deduplicate a troublesome candidate until its score improves. Keep raw rows and prove fused-observation parity when reproducing an existing result.

When using the external ReID service, keep feature cadence, local ReID, Kafka transport and other settings matched for on/off comparisons. Start from an isolated empty service and topic for independent recordings. Readiness alone does not prove ingestion. Check embedding counts and actual searches, and never reset a shared service.
