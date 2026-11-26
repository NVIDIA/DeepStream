# Diagnostics and visual review

## Establish what is observable

Without GT, do not label a proxy score as tracking accuracy. Long tracks can be wrong identities. Fewer IDs can mean false merges. Smooth positions can be stale predictions. Fewer jumps can mean that difficult people were suppressed. Compare observation coverage and visual misses alongside continuity.

Keep detector boxes, tracker boxes, online IDs, feet/world positions, target states, confidence/visibility, source timestamps and camera identity where available. Distinguish detector-supported tracks from prediction-only, see-through and retrospective exports. Record each field's producer. Do not synthesize observations and then present them as tracker output.

Validate exported values, not just enabled-looking YAML. Visibility belongs to TargetManagement.outputVisibility, while foot/hull export belongs to ObjectModelProjection. A misplaced visibility key can leave every exported value at -1. The basic overlay enables valid visibility and foot metadata, and removes that known misplaced key. Verify production exports after diagnostics are disabled.

## Read existing instrumentation first

Inspect the selected source revision's module destructors, reporting functions and debug-level guards. Existing public code includes aggregate DataAssociator gate/pass and rescue counters, TargetManagement termination and shadow-age counters, and trajectory/multi-view diagnostics. Verify the exact log labels in source before parsing them. Do not assume every debug level or counter exists in a prebuilt library with the same SDK version label.

Useful questions for instrumentation:

- At which gate are otherwise plausible associations rejected: IoU, size, world distance, appearance or total score?
- Are true tracks lost during probation, shadowing, peer handoff or re-association?
- Do false positives predominantly come from young tentative tracks, persistent shadow tracks, duplicate births or see-through predictions?
- How many eligible peer comparisons have enough common observations, and how stale are they?
- How often is an ID adopted, corrected or re-associated, and what competing candidates and score margins existed?
- Are invalid/negative-depth projections or inconsistent model heights concentrated in particular cameras?
- Is the effective measurement-noise matrix finite, symmetric and positive semidefinite at the Kalman update? Trace detector, visual-tracker and peer inputs separately, including whether a placeholder covariance is replaced before becoming measurement noise. Source inspection alone does not prove that a suspect path runs.
- Are ReID queries backed by useful embeddings and actual gallery ingestion? Which queries adopt a different identity, and are those adoptions correct on images?

Counters are evidence of branches taken, not proof a decision was correct. Preserve aggregate rates with denominators. Use bounded traces around flagged times instead of logging every pair across a day-long run. Guard new counters and traces, use atomics or correct per-thread aggregation, and keep algorithms unchanged. Test dynamic parameter updates if instrumentation touches config handling.

## Proxy reports

`evaluate.py diagnostics` reports observations, camera-local tracks, short tracks, gaps, duplicates, high-speed intervals and cross-camera disagreement, with eligible pair counts and flag rates. Select --class-id when applying physical thresholds to mixed-class inputs. Choose thresholds from class, site scale and camera geometry, never apply a person threshold to forklifts. For variable frame rates or missing frames, normalize a timestamp-based axis or calculate physical deltas directly in a custom adapter.

Compare rates and denominators, not just counts: more correctly shared IDs create more cross-camera comparisons and can increase the number of flagged pairs. Shared-ID disagreement is conditional on already sharing an ID, so it does not measure missed associations or overall ID accuracy. An empty denominator or unset threshold is reported as null, not a perfect score.

Also review detection-to-track assignment coverage, camera entry/exit counts, overlap identity agreement, apparent ID changes across occlusion, occupancy and impossible simultaneous locations. The bundled helper does not identify every semantic ID switch without labels. Add a targeted diagnostic when the data supports it and explain its limits.

Select a fixed, diverse review set before tuning. Include uniform samples plus crossings, partial occlusions, coverage transitions, image edges, distant objects, crowded and empty periods. Add candidate-triggered anomaly windows, but retain the fixed review set so the agent cannot improve scores by selecting easier images.

## Inspect images and short sequences

Use `review_frames.py` to render simultaneous raw camera views with canonical boxes and IDs. Open the generated images with the available image-reading tool. A filename or script exit status is not visual inspection. Verify that the rendered frame agrees with the source video and any annotation mapping. Missing/undecodable frames must be shown as unavailable, not replaced with an earlier frame.

For suspected identity errors, inspect several frames before, during and after the event. A single frame cannot establish continuity or a correct re-association. Keep consistent colors for IDs and include camera names and absolute source time when known. Do not judge an edited/smoothed demo as the quantitative output.

For geometry issues, reuse the sample's camera projection and visualization utilities to overlay recorded feet/cylinders and show a BEV map. State whether geometry is recorded tracker output or a projection using configured class dimensions. Do not substitute GT, guessed object dimensions or detector boxes for recorded 3D state. Compare cameras that see the feet clearly with occluded views. Check the raw fused trail as well as each camera estimate.

When world positions jump while boxes and raw video remain smooth, inspect clipping/alignment branches before changing association or noise thresholds. Pixel-boundary heuristics can switch projection reference points abruptly. Record consecutive boxes, measurements and branch conditions, and distinguish a projection discontinuity from calibration error. Do not hide it with smoothing or fit unknown object dimensions to the desired path.

Use a compact review record per window: camera IDs, exact frames/source times, what is visible, apparent misses/duplicates/ID changes, geometry alignment, confidence, and the next test. Compare baseline and candidate with the same scale and overlays. If possible, hide candidate names during review to reduce preference for the expected winner.

## Promotion without GT

Require consistent improvements across the fixed visual windows and diagnostic evidence, without obvious losses of observation coverage. Do not promote a candidate merely because it minimizes ID count, speed, jitter or fragmentation. Describe conclusions as qualitative or proxy-supported. If identity correctness remains ambiguous, request a small human-labeled validation subset rather than inventing identity GT. Keep tuning videos separate from any later holdout annotation.

## Final build

Turn off temporary logging and instrumentation, rerun the selected config and compare completion, output quality and performance. Timing from an instrumented build does not establish production throughput. Keep the selected deployed communication mode: same-instance memory experiments do not validate asynchronous MQTT behavior. Do not leave a diagnostic shared library installed over the user's normal library.

For multiple tracker instances, verify a collision-free global identity namespace using full unique IDs or explicit nonoverlapping targetIDOffset ranges. Local counters with useUniqueID off can start at zero independently in each process. Native dumps that mask IDs to 16 bits cannot establish distributed identity correctness by themselves.
