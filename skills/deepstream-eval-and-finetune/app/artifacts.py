# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact-path resolution shared by the Web UI server and pipeline."""

import json
import os


def artifact_dir_candidates(preset, field):
    """Return the canonical directory followed by declared read-only aliases.

    Presets are user-editable (custom runs are written into the registry at runtime), so the
    aliases value is not guaranteed to be a list. Normalise defensively rather than unpacking
    blind: a bare string would otherwise splat into one candidate *per character*, and a number
    would raise TypeError deep inside a request handler. Anything unusable is ignored -- a
    malformed alias must never take the UI down or invent bogus paths.
    """
    aliases = preset.get(f"{field}_aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    elif not isinstance(aliases, (list, tuple)):
        aliases = []
    candidates = [preset[field], *(a for a in aliases if isinstance(a, str))]
    return list(dict.fromkeys(p for p in candidates if p))


def _eval_target_class_count(root, candidate):
    """Number of KPI categories the eval set in `candidate` was built for, or None if unknown.

    `ground_truth.json`'s `category_map` maps the *dataset's* category ids onto model labels, so
    its size identifies the workload's target label space (aerial-sheep 1, PCB 6) even when the
    deployed engine is the same 80-class stock model for both.
    """
    gt = os.path.join(root, candidate, "eval", "eval_set", "ground_truth.json")
    try:
        with open(gt) as fh:
            return len(json.load(fh).get("category_map") or {})
    except (OSError, ValueError):
        return None


def _alias_matches_workload(root, preset, candidate):
    """Reject an alias whose eval set targets a different workload's label space.

    Built-in demos can share a base model (both RT-DETR presets use PekingU/rtdetr_r50vd), so a
    model-derived alias like `models/rtdetr_r50vd` is ambiguous between them: whichever workload
    ran last owns the directory. Adopting it blindly would report one demo's baseline under the
    other -- e.g. aerial-sheep's non-zero mAP shown as PCB's, whose honest baseline is 0.
    Fail open when either side is unknown, so this only ever rejects a proven mismatch.
    """
    expected = preset.get("ft_nclasses")
    found = _eval_target_class_count(root, candidate)
    if expected is None or found is None or found == 0:
        return True
    return int(found) == int(expected)


def resolve_artifact_dir(root, preset, field, sentinel):
    """Select the first complete artifact directory, falling back to the canonical write path.

    Aliases are adopted only when ``sentinel`` exists AND the directory's eval set targets this
    preset's label space. The sentinel check prevents a partial CLI deploy from appearing ready;
    the label-space check prevents a sibling demo's baseline from being adopted as this one's.
    The canonical ``preset[field]`` is always trusted -- the UI writes it itself. Writers and
    delete/clear operations must continue to use the canonical path.
    """
    canonical = preset[field]
    for candidate in artifact_dir_candidates(preset, field):
        if not os.path.isfile(os.path.join(root, candidate, sentinel)):
            continue
        if candidate != canonical and not _alias_matches_workload(root, preset, candidate):
            continue
        return candidate
    return canonical


def resolve_orig_dir(root, preset):
    """Resolve a complete deployed-baseline directory for UI reads and reports."""
    return resolve_artifact_dir(
        root, preset, "orig_dir", os.path.join("eval", "engine_metrics.json")
    )
