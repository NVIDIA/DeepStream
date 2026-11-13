---
name: deepstream-tune-mv3dt
description: Tune DeepStream multi-view 3D tracking pipelines on user recordings, with ground-truth evaluation or diagnostics and visual review. Use for configuration sweeps, identity continuity, missed or duplicate tracks, unstable geometry, or accuracy and throughput tradeoffs. Inspect and adapt supplied data, ask for a dataset-scaled long-running budget, begin with geometric tracking, and incrementally test optional features.
---

# Tune MV3DT

Optimize the user's pipeline through measured configuration experiments. Use the public tracker source to understand active parameters and add temporary diagnostics when necessary. Deliver ordinary DeepStream configs that run without the tuning harness.

Use the existing [MV3DT run skill](../deepstream-run-mv3dt/SKILL.md) for setup and launch details. This skill owns experiment selection, comparisons, budget, and final validation. Do not execute a single-run workflow over the campaign directory.

## Inspect inputs and agree on a budget

1. Discover recordings, detector/model files, calibration, existing configs, optional detection caches, and GT. Ask only for facts that cannot be determined from these files. Read [inputs and evaluation](references/inputs-and-evaluation.md) before conversion.
2. Probe every video. Confirm camera order, overlapping time ranges, source and processed frame counts, image transforms, timestamp meaning, calibration units, object classes, and GT coverage. Reformat into a new workspace, retaining the originals and a reversible mapping. Capture RTSP once for repeatable offline tuning only when recording is authorized.
3. Use `scripts/campaign.py inspect` to propose a budget with a dimensionless exploration multiplier of 60, keeping recording duration and budget in the same time units:

   `budget_seconds = camera_count × synchronized_recording_seconds × 60`

   Store seconds internally and divide by 3600 only to display hours. For 12 cameras and two minutes: 12 × 120 × 60 = 86,400 seconds = 24 hours. For multiple recordings, sum each recording's camera-seconds before multiplying by 60. Do not multiply by camera count again when summing individual video durations. Use physical recording time and the selected synchronized interval, not accelerated playback or unrelated portions of longer files. Unknown duration requires a recording interval before calculating a budget.
4. Present camera count, selected duration, proposed hours, GPU/resources, and outputs. Ask: “The proposed tuning budget is 24 hours for 12 cameras × 2 minutes. Is that OK, or would you like a different budget?” Replace the numbers with inspected values. If the user already supplied a budget, use it without asking again. Silence is not acceptance.
5. Record the accepted budget with `approve`. This is an exploration allowance, not a runtime estimate. Start its clock at approval and reserve time for full-sequence confirmation and clean-config validation. A resume retains the deadline. Ask before extending it. Do not install a cron job or restart an expired campaign implicitly.

## Freeze the comparison

Record hashes for inputs, GT, calibration, detector/cache, evaluator, source revision and actual loaded tracker library. Confirm a rebuilt library is used through `ll-lib-file`, not just that a build succeeded. Keep launch mode, GPU assignment, communication topology, stream order, frame mapping and output semantics fixed between candidates.

Use HOTA as the default primary objective with GT. Report MOTA and IDF1 beside it and retain nondominated alternatives. Explain tradeoffs and honor user-specified FPS, latency, FP, FN or identity requirements. Do not select by a weighted score whose priorities were not agreed.

Distinguish the best tested candidate from a deployment-ready result. Preserve negative metrics and visible failures, and do not describe a relative improvement as acceptable accuracy without an agreed quality target.

Freeze evaluation before exploration. A score increase obtained by dropping difficult GT, shortening annotation coverage, changing matching tolerance or fusion, or editing GT is not a tracker improvement. Keep benchmark GT read-only. GT corrections start a separate, versioned evaluation.

Without GT, report proxy diagnostics and visual findings, not HOTA/MOTA/IDF1. Read [diagnostics and inspection](references/diagnostics-and-inspection.md). Shorter or smoother output alone does not establish better tracking.

## Tune in levels

Read [parameter priorities](references/parameter-priorities.md). Inspect parsing and call sites in the selected tracker revision before assigning ranges. Check whether a parameter is active, overridden, clamped, class-specific or dependent on another module. Do not inherit archived tuned values as defaults.

| Level | Explore | Keep fixed |
|---|---|---|
| 0 | Input validity, detector quality and basic geometry | Dataset, calibration convention, GT coverage |
| 1 | Basic geometric MV3DT | Visual tracker, pose, ReID/service, re-association, ID correction and see-through off |
| 2 | Detection acceptance, data association and target lifecycle | Optional modules off |
| 3 | Initial multi-view ID acquisition, peer matching and geometry | Optional modules off, known peer topology |
| 4 | Visual tracker, pose, local ReID, re-association and see-through | One feature at a time, then justified interactions |
| 5 | External ReID service and ID correction | Matched local settings, isolated service state |
| 6 | Repeats, held-out footage, deployment mode and clean export | Selected config and frozen evaluation |

Start from the published NvSORT/basic tracker config, add only the MV3DT geometry and communication sections required by the deployment, and apply `assets/basic-overlay.yml`. Keep initial multi-view ID acquisition enabled. Validate the generated config because auto-configurators can enable modules when converting 2D configs to MV3DT. Keep the user's original pipeline as a comparison, but explore from the basic baseline.

Cache actual detector outputs for tracker sweeps when supported, preserving boxes, scores, classes and frame correspondence. Lowering a detector threshold below an already filtered cache cannot recover missing detections. Detector changes require new inference and a new cache lineage. Replay does not remove pose/ReID inference costs.

Use diagnostics to propose a small related group of changes, record the hypothesis, then compare with the current reference. Start with coarse ranges, refine useful regions and test interactions. Do not perform an unrestricted Cartesian sweep. Use a fresh directory per run and retain failures and rejected candidates.

Use representative windows for triage, covering occlusion, overlap transitions, small objects, crossings, empty scenes and coverage gaps. Warm up tracking before scoring a window. Promote only after full-sequence evaluation. Repeat finalists at least three times with the same launch protocol, retaining mean and spread. A single exceptional scheduling outcome is not a stable improvement.

Continue useful exploration across levels within the approved budget. Stop early only when the agreed target is met, an input/infrastructure problem prevents valid comparison, or the evidence no longer supports another useful experiment. Report why. A short initial sweep is not the end of a long-running campaign while promising hypotheses remain.

## Run and observe

Read [harness commands](references/harness.md). Use `scripts/campaign.py` for inspection, deadline enforcement, isolated candidates, run records, summaries and clean config export. Use `scripts/evaluate.py` for canonical world-space GT evaluation with the sample TrackEval metric core, and for no-GT trajectory diagnostics. Use `scripts/review_frames.py` for raw video contact sheets with canonical boxes and IDs.

Write dataset-specific adapters in the campaign workspace for unsupported formats. Test camera IDs, class IDs, timestamps, units, box conventions and known rows before launching. Reuse the public sample configurator and evaluator where applicable. Do not require internal repositories, private images, workstation paths or an internal build bridge.

Run candidates in a foreground process, with campaign and shared GPU locks. A checkpoint is not permission to start a duplicate worker. Inspect active jobs on resume. Stop only the process group or uniquely named container owned by this campaign. Do not reset shared ReID data or alter user services.

During long work, report completed trials, elapsed/remaining budget, current metrics and tradeoffs, failures, and the next hypothesis. Use product-supported background execution and monitoring when available. Keep machine-readable state and logs so another session can resume the search.

Enable diagnostics selectively. Start with aggregate counters and low debug levels. Add bounded, thread-safe instrumentation in an isolated source worktree only when it answers a specific question. Compare with the unmodified baseline to detect instrumentation effects on output or scheduling. Algorithm changes need a separately requested mode.

## Finish with runnable configs

Export the selected config and simple baseline. Disable temporary `debugVerboseLevel`, `saveLog2File` and diagnostic dump flags. Retain required production outputs, including embeddings, visibility, feet and geometry. Document model locations.

Run the exported clean config again, collecting evaluation data through a separate capture/export adapter where needed. Confirm quality, completion and resource use. Measure live detector-inclusive pipeline FPS with all cameras concurrent, separately from replay throughput, engine builds and service initialization. Distinguish online IDs from retrospective trajectory-file identities.

Deliver configs, before/after metrics or visual evidence, tradeoffs, selected run IDs, exact commands, model/runtime/source hashes and limitations. Identify unvalidated modes/classes. Keep instrumentation as a separate patch, disabled or absent from the deployment build. Do not stage or commit without the user's instruction. Never push.

Documentation: the [tracker](https://docs.nvidia.com/deepstream/9.2/text/DS_plugin_gst-nvtracker) and [MV3DT](https://docs.nvidia.com/deepstream/9.2/text/DS_MV3DT) pages. The same pages are in this repo as `docs/text/DS_plugin_gst-nvtracker.mdx` and `docs/text/DS_MV3DT.mdx`. Use `deepstream-run-mv3dt` to launch the sample and custom-dataset runs.
