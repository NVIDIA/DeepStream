# Parameter priorities

Treat this as a search order, not a universal winning preset. Confirm each key in the selected public source and inspect its enabled call path. The public baseline configs and the deployment's effective generated config are authoritative for availability.

## Critical first

| Module | Parameters | Evidence and direction |
|---|---|---|
| Detector and BaseConfig | Detector confidence/NMS, `minDetectorConfidence`, `checkClassMatch` | Review missed small objects and false detections before tracking. Keep classes fixed. Higher thresholds trade recall for precision. Do not raise confidence merely to hide poor calibration. |
| TargetManagement | `probationAge`, `earlyTerminationAge`, `maxShadowTrackingAge`, `maxTargetsPerStream` | Short tentative false tracks suggest longer confirmation or earlier termination. Fragmentation during occlusion suggests longer shadow retention, but verify no stale/ghost tracks. Capacity must include shadow targets. |
| TargetManagement | `minIouDiff4NewTarget` | Suppresses new detections overlapping existing targets. A higher threshold permits more overlapping new tracks, so inspect duplicates and crowded-scene misses. |
| DataAssociator | `minMatchingScore4Iou`, `minMatchingScore4SizeSimilarity`, `minMatchingScore4Overall` | Inspect rejection counts and near-threshold pairs. Relax only the gate implicated in losses. Strong perspective and partial occlusion can distort size similarity. |
| DataAssociator | `tentativeDetectorConfidence`, `minMatchingScore4TentativeIou` | Inspect low-confidence detections separately from established tracks. Check fast motion and detector cadence before demanding high IoU. Keep usePrediction4Assoc off for the current geometric estimators, as explained below. |
| MultiViewAssociator | `minPeerTrackletMatchScore`, `minCommonFrames4MatchScore`, `maxPeerTrackletSize` | Verify common visibility and sufficient history. Lower requirements can improve handoff but admit unrelated people. Longer history costs memory and can mix stale observations. |
| DataAssociator | `minMatchingScore4PeerAssocIou` | Not a general peer matching threshold. It rescues an unmatched low-confidence detection, normally dropped, as a new-target candidate when it overlaps the projected position of a recently updated peer target by more than this IOU. A value of 1.0 or more disables it, and it acts only with the MultiViewAssociator enabled. Inspect cross-view recovery and ghost boxes, not only the final fused trajectory. |

A useful first experiment changes one threshold family with a few values around the current setting. Inspect FP/FN and IDSW changes to distinguish recall, duplicate suppression and identity effects. Sweep related lifetime settings in seconds, then convert to the actual tracker cadence. Do not blindly scale confidence or similarity thresholds with FPS.

Check effective clamps before choosing values. In the reviewed source, maxShadowTrackingAge is raised to at least probationAge + 10. minCommonFrames4MatchScore and maxPeerFrameDiff4NoDet are at least 2. Different requested values can therefore produce identical effective behavior.

## Secondary controls

| Area | Parameters | Conditions |
|---|---|---|
| Association weights | `matchingScoreWeight4Iou`, `matchingScoreWeight4SizeSimilarity` | Keep visual/ReID weights zero while those features are off. Validate score scale when changing weights and overall threshold together. |
| Motion model | `processNoiseVar4Loc`, `processNoiseVar4Vel`, applicable state-estimator settings | Match actual units and time evolution. Noise values are not directly portable between meters, grid coordinates and pixels. Check innovation and velocity diagnostics. |
| World-space association | `maxWorldDist4PrimaryAssoc`, `maxWorldDist4QuasiactiveRescue`, `maxWorldDist4SizeBypass`, `maxWorldDist4ActiveIouRescue` | Read activation and threshold logic. Tighter world gates require reliable calibration and feet estimates. Inspect source-specific valid-coordinate assumptions. |
| Peer freshness | `maxPeerFrameDiff4NoDet`, `recentlyActiveAge`, `maxTrackletMatchingTimeSearchRange` | Convert to time only after confirming whether the code counts peer, ego or batch frames. Do not use matching search to conceal capture-clock drift. |
| Geometry | `enableBboxUnClipping`, configured class height/radius, `minPeerVisibility4Fusion` | Review image-border truncation, feet occlusion and object dimensions. Preserve independent calibration. Person cylinders do not transfer to forklifts without appropriate dimensions. |
| Peer fusion | `maxPeerToPredDistance4Fusion`, `enableSeeThrough` | In the reviewed type 4 implementation, maxPeerToPredDistance4Fusion is overridden to 100 world units. A sweep of the configured value would be ineffective. Verify the selected revision before including it. |

Fusion activation differs by frame type. With a matched detection, a peer measurement must also pass the visibility and camera-distance comparisons against the ego detection. Without one, those two checks are skipped. Peer freshness applies to both, and type 4 keeps its fixed 100 world unit distance gate in both cases. A tuning change can therefore affect detection frames and non-detection frames differently, so check them separately.

Visibility, unclipping and see-through affect projected boxes and apparent detections in cameras with occlusion. Inspect raw per-camera views and a fused map together. Better-looking cylinders can result from incorrect height or camera placement, so do not judge only by overlays.

## Optional features, one at a time

1. **Visual tracker:** enable `visualTrackerType: 2` (VPI) only after the geometric baseline is stable. Start with a type 3 estimator control, since type 4 with visual tracking is unvalidated in the reviewed source, as explained below. Explore feature size, search region and applicable confidence thresholds. Account for image resolution, object pixel size, detector interval and additional compute. `minTrackerConfidence` is not an equally meaningful knob with every visual tracker disabled.
2. **Pose:** enable the supported estimator, first for initialization. Explore `ObjectModelProjection.minPoseConfidence` and `PoseEstimator.poseInferenceInterval`. In the reviewed implementation, -1 still runs during tentative/unassigned-ID states. Test crouching, children, occlusion and initialization failures. Keep VPI crop settings compatible with input dimensions.
3. **Local ReID:** enable the desired `reidType` and verify model preprocessing, dimensions, normalization and class eligibility. Explore `TrajectoryManagement.reidExtractionInterval`, `reidHistorySize`, `minVisibility4GalleryUpdate` and applicable appearance association weights. Type 2 primarily supports re-association, not automatically all online association stages. The current factory rejects types 1 and 3 with NvDCF/VPI visual tracking, while type 2 is compatible. Verify this before testing combinations.
4. **Local/peer re-association:** distinguish `TrajectoryManagement.enableReAssoc`, `enable3DReAssoc`, `MultiViewAssociator.enableLatePeerReAssoc` and `enablePeerReidEmbedding`. Inspect actual dependencies. Explore gap lengths, archive ages, world distances and appearance thresholds after observing failure windows. The reviewed world-distance re-association path doubles maxFrameGap4WorldDistReAssoc, treats nonpositive values as no temporal cap, and requires maxWorldDist4ArchiveReAssoc > 0. Verify peer history covers the intended window and recheck current source before copying values.
5. **See-through and ID correction:** test independently. `MultiViewAssociator.enableIDCorrection` drops the newer of two same-ID targets whose tracklets stop matching, while `ReIDService.enableIDCorrection` drops a local ID that the external service keeps giving to another target. They are separate switches, so compare them separately. They can recover continuity but also generate wrong projections or propagate a mistaken identity. Keep the winning configuration only after full-sequence and image checks.
6. **External ReID service:** compare remote queries off/on while holding local features and cadence fixed. Explore `minVisibility4ReidServiceQuery`, `minTopClusterConfidence`, `minSecondClusterConfidence`, `maxClusterVariance`, `topK`, `maxGroupCount` and blocking behavior when supported. Record real unequal-ID adoptions and errors, not only successful HTTP requests. An IDF1 gain can come with lower MOTA or throughput.

After feature tests, explore limited interactions between the features that helped. Also compare each finalist with the simple configuration, since adding every module can reduce accuracy or throughput.

## Version-specific integration checks

Verify these current-source limitations before exploring optional combinations. They are not threshold-tuning opportunities:

- Treat stateEstimatorType 4 with visualTrackerType 1 or 2 as unvalidated. In reviewed tracker tree 1e63caeb05c5, NvTrackedObject.cpp queues visual localizations with a negative identity placeholder covariance and isDetectorInput false. SimpleEstimator2::update copies that covariance directly into measurement noise for type 4, despite its comment promising a default-noise fallback. Type 3 instead selects measurementNoiseVar4Tracker. This is a source-only concern, not a reproduced runtime failure. Detector and peer measurements in the 3D path carry computed covariances, so this finding does not by itself invalidate the visual-tracker-off type 4 baseline.
- Geometric estimators 3/4 predict two world measurements, while the ReID1/3 Mahalanobis path in ReID.cpp compares against four pixel-box measurements. Use type2 for geometric local-ReID exploration unless that integration is fixed and independently validated. Disabling the later Mahalanobis gate does not remove the computation.
- The image-space trajectory projector in NvTrajectoryManager.cpp uses four observations. With stateEstimatorType4, SimpleEstimator2's vector update assigns a 2×2 measurement covariance. Keep that image-space re-association route on a validated type3 configuration or leave it off, pending a separate source fix.
- Audit pose exports independently from the footprint used for tracking. In the reviewed PoseInference.cpp, setPose3d receives the pose25d buffer. Do not use that affected joint export as valid metric 3D pose or evidence of off-ground tracking. The separate pose25d footprint path needs its own validation.

Compare estimator types 3 and 4 with visual tracking off first. If investigating type 4 with visual tracking, retain a type 3 control with the same visual tracker and verify the actual measurement covariance path before promoting the combination. Label the experiment diagnostic until validated. Switching estimator type also changes effective peer-distance gating, so it is not automatically a noise-model-only ablation.

## Mixed-class prerequisites

- Gate human BodyPose3D with PoseEstimator.operateOnClassIds using the detector's actual person class. An empty list means all classes, not disabled inference. Do not apply human keypoint geometry to forklifts.
- The reviewed local ReID implementation has no operateOnClassIds filter. Do not invent that key or assume ensemble models route by class: enabled ensemble models process the same object batches. Validate the model's class scope before enabling appearance-based decisions.
- Selected ensemble models inherit common ReID settings and then apply model overrides. A selected submodel with reidType0 can still initialize its model/crop runtime. Disable the whole module with the root reidType0 when establishing the basic baseline.
- reidExtractionInterval controls type 2's post-update extraction path. Types 1 and 3 infer detection features before association, so changing that interval does not establish their inference cadence.
- ReIDService has its own operateOnClassIds gate, but its query payload does not include a class filter. Verify gallery ingestion and model/class isolation separately. A human-only query gate does not automatically create a human-only gallery.
- Start service comparisons with one explicit embedding model. Ensemble export and the target feature consumed by service queries can have different producers. Prove matching model spaces before treating an ensemble/service experiment as valid.
- checkClassMatch protects ordinary association, but the reviewed archive and 3D re-association paths lack equivalent class checks. Keep those optional modes outside a validated mixed-class deployment until a separately authorized source fix or genuinely class-isolated pipeline establishes safety.

## Avoid inactive or misleading searches

- Read parser defaults, bounds and use sites. A missing key may silently select a materially different mode.
- Type 4 uses projected per-measurement covariance. Fixed measurement noise settings used by older estimators are not equivalent controls.
- A 2D tracker converted by a configurator can acquire visual tracking or another estimator automatically. Verify the final YAML, not only the requested overlay.
- NvSORT's positive noiseWeightVar4Loc/Vel enables covariance scaled by 2D box height. Do not carry it into geometric type 3/4, whose measurements lack that height element. The basic overlay restores the disabled source defaults (-0.1 for both). Zero also disables this path but causes fallback warnings. Select fixed world-noise settings in the dataset's units rather than importing pixel-space values.
- NvSORT also enables usePrediction4Assoc. In the reviewed implementation this accessor interprets predicted state as a 2D box, not world XY and velocity. Keep it off with geometric types 3/4 to use the projected tracker box, unless a later source revision implements geometric prediction correctly. This is a valid-baseline prerequisite, not an accuracy gain from a fair parameter comparison.
- Check crop alignment when enabling pose or ReID without visual tracking. Legacy cropping (useVPICropScaler 0) requires tracker dimensions divisible by 32. Prefer a supported VPI crop path with unchanged geometry before resizing. A tracker-resolution change requires a new verified pixel-coordinate/calibration contract, not an incidental tuning adjustment.
- Some world-rescue and 3D re-association paths treat negative world X as unavailable. Do not change the dataset origin or filter GT silently to activate them. Record the limitation and request a separate geometry or source change when needed.
- Increasing shadow duration or enabling backprojection can reduce fragmentation proxies while adding false tracks.
- Raising dump-age or visibility filters can improve exported scores without improving online tracking. Freeze output semantics.
- A detector cache may already remove classes, low confidence boxes, or distant objects. Log these admission rules and never compare it as if it were unfiltered live detection.
- Do not sweep zero-weight similarity gates as though their model contributed useful evidence. Verify the effective score computation.

Documentation: [State Estimation](https://docs.nvidia.com/deepstream/9.2/text/DS_plugin_gst-nvtracker#state-estimation) and [External ReID Service](https://docs.nvidia.com/deepstream/9.2/text/DS_plugin_gst-nvtracker#external-reid-service) in the tracker documentation, and [Multi-View Measurement Fusion](https://docs.nvidia.com/deepstream/9.2/text/DS_MV3DT#multi-view-measurement-fusion) in the MV3DT documentation. The same pages are in this repo as `docs/text/DS_plugin_gst-nvtracker.mdx` and `docs/text/DS_MV3DT.mdx`.

Source starting points: the public [tracker documentation](https://docs.nvidia.com/deepstream/9.2/text/DS_plugin_gst-nvtracker), `src/utils/nvmultiobjecttracker/config/`, and the matching module implementations under `src/utils/nvmultiobjecttracker/src/modules/`.
