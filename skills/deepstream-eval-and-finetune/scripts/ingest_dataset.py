# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ingest a detection dataset into COCO — once, for both skills.

Some HF detection datasets are NOT loadable as a `dataset_id`:
  * a YOLO-zip with a broken viewer (e.g. the Military-Aircraft set ships one
    `dataset.zip` with a YOLO layout), OR
  * a SCRIPT-based dataset (a `*.py` loader, which `datasets` >=3 rejects with
    "Dataset scripts are no longer supported") that bundles its data as Roboflow
    COCO-export zips — e.g. `keremberke/aerial-sheep-object-detection` ships
    `data/{train,valid,test}.zip`, each holding `_annotations.coco.json` + images.

This script turns either into a COCO `annotations_<split>.json` + a `labels.txt`,
which feed BOTH:
  * `tao-finetune-huggingface-model`  via `local_dataset_format: coco`
  * this skill's `build_eval_set.py --local-coco`

Layouts handled inside the (extracted) root:
  * YOLO:  images/<split>/*.jpg + labels/<split>/*.txt  (normalized cx cy w h)
  * COCO:  a `_annotations.coco.json` (+ images) per split — used as-is.
Empty YOLO label file == background image (kept, no annotations).

Class NAMES come from (in order): --class-names file > data.yaml/obj.names in the
root > integer ids as a last resort. The model `labels.txt` must list classes in
this same id order so downstream label indices line up.
"""
import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

# Conventional SMALL subdirs where dataset exports keep zips/config/splits. We probe these by
# EXACT name (a single stat each) instead of '*/' globs, which would list every depth-1 dir —
# including image folders of 10k+ files — and crawl for minutes over a network (SMB) mount.
_CONFIG_SUBDIRS = ("data", "raw", "dataset", "extracted")


# Zip-bomb guards: refuse archives that would exhaust disk. Generous enough for real datasets
# (a ~13 GB PCB export unpacks fine) but far below a decompression bomb (which expands to TB-PB).
_MAX_ZIP_MEMBERS = 500_000
_MAX_ZIP_UNCOMPRESSED = 64 * 1024 ** 3        # 64 GiB total extracted
_MAX_MEMBER_RATIO = 200                       # per-file compression ratio (catches a single huge bomb)


def _safe_extract(zf, dest):
    """Zip-Slip- AND zip-bomb-safe extraction of an UNTRUSTED archive (local/HTTP/HF source).

    - Zip Slip (CWE-22): a crafted member ('../../x' or an absolute path) could escape `dest` and
      overwrite arbitrary files. Only members whose resolved path stays inside `dest` are extracted;
      escapes are skipped with a warning.
    - Zip bomb (CWE-409): a small archive can expand to exhaust disk. Refuse archives over
      _MAX_ZIP_MEMBERS members or _MAX_ZIP_UNCOMPRESSED total uncompressed bytes, and skip any single
      member whose compression ratio is implausibly high.
    Legitimate dataset zips (plain relative paths, already-compressed JPEGs) pass unchanged."""
    dest = Path(dest).resolve()
    infos = zf.infolist()
    if len(infos) > _MAX_ZIP_MEMBERS:
        raise ValueError(f"refusing zip with {len(infos)} members (> {_MAX_ZIP_MEMBERS}; possible zip bomb)")
    total = sum(i.file_size for i in infos)
    if total > _MAX_ZIP_UNCOMPRESSED:
        raise ValueError(f"refusing zip: expands to {total / 1024 ** 3:.1f} GiB "
                         f"(> {_MAX_ZIP_UNCOMPRESSED // 1024 ** 3} GiB; possible zip bomb)")
    for member in infos:
        if (member.compress_size > 0 and member.file_size > 100 * 1024 ** 2
                and member.file_size / member.compress_size > _MAX_MEMBER_RATIO):
            print(f"[ingest] SKIP suspicious high-ratio member (possible bomb): {member.filename}")
            continue
        target = (dest / member.filename).resolve()
        if target != dest and dest not in target.parents:
            print(f"[ingest] SKIP unsafe zip member (path escapes {dest.name}/): {member.filename}")
            continue
        zf.extract(member, dest)


def _extract_zips(local, dest):
    """Extract any *.zip under `local` into dest/extracted/<stem> (one subdir each, so
    multi-zip Roboflow exports don't overwrite one another); return the extracted root,
    or `local` itself if it has no zips (already an unpacked layout)."""
    local = Path(local)
    # Root-level zips + zips in conventional SMALL subdirs only. NEVER use a '*/' glob over a
    # large/network-mounted tree: '*/' lists the contents of EVERY depth-1 subdir, so it
    # enumerates image dirs (10k+ files) — measured ~112s/op over SMB. Dataset-export zips live
    # at the root or in a data/-style subdir (e.g. data/train.zip), so probe those by exact name.
    zips = sorted(local.glob("*.zip"))
    for sub in _CONFIG_SUBDIRS:
        d = local / sub
        if d.is_dir():
            zips += sorted(d.glob("*.zip"))
    if not zips:
        return local
    ex = Path(dest) / "extracted"
    ex.mkdir(parents=True, exist_ok=True)
    for z in zips:
        sub = ex / z.stem
        sub.mkdir(parents=True, exist_ok=True)
        print(f"[ingest] extracting {z.name} -> {sub}")
        with zipfile.ZipFile(z) as zf:
            _safe_extract(zf, sub)
    return ex


def _hf_repo_from_url(url):
    """'https://huggingface.co/datasets/ns/name/tree/main?x=1' -> 'ns/name'."""
    u = re.sub(r"[?#].*$", "", url.strip())
    u = re.sub(r"^https?://(www\.)?huggingface\.co/", "", u)
    u = re.sub(r"^datasets/", "", u)
    u = re.sub(r"/(tree|blob|resolve)/.*$", "", u)
    return u.strip("/")


def _hf_download(repo_id, dest, revision="main"):
    from huggingface_hub import snapshot_download
    raw = Path(dest) / "raw"
    print(f"[ingest] downloading HF dataset {repo_id} -> {raw} (revision={revision})")
    local = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(raw),
                              revision=revision)
    return _extract_zips(local, dest)


def _maybe_download_and_extract(source, dest, revision="main"):
    """Resolve a dataset SOURCE to an extracted root directory. Accepts any of:
      * local directory  (a YOLO/COCO layout, or a dir containing zips)
      * local .zip file
      * http(s) URL to a .zip
      * HuggingFace dataset URL (https://huggingface.co/datasets/ns/name)
      * HuggingFace repo id (ns/name)
    """
    src = str(source).strip()
    p = Path(src)

    # A clearly-local path (absolute, ./, ../, ~, file://) that doesn't exist is almost always a
    # volume that isn't bind-mounted into the container — fail loudly with the fix, instead of
    # silently misfiring as an HF repo id (which raises a confusing HFValidationError).
    if src.startswith(("/", "./", "../", "~", "file://")) and not p.exists():
        sys.exit(f"ERROR: local dataset path not found inside this container: '{src}'. "
                 f"Bind-mount it at the SAME path — e.g. launch the UI with "
                 f"DATASET_MOUNT={p.parent if p.parent != p else src} bash .../app/launch.sh, "
                 f"or add -v {src}:{src}:ro to the docker run.")

    # 1) local path (directory or .zip) — no download
    if p.exists():
        cache_root = os.environ.get("DS_DATA_CACHE_ROOT")
        if cache_root:
            from dataset_cache import stage_local_source
            refresh = os.environ.get("DS_DATA_CACHE_REFRESH", "").lower() in (
                "1", "true", "yes", "on"
            )
            original = p
            p = stage_local_source(p, cache_root, refresh=refresh)
            print(f"[ingest] Linux dataset cache: {original} -> {p}"
                  f"{' (refreshed)' if refresh else ''}")
        if p.is_dir():
            print(f"[ingest] using local directory {p}")
            return _extract_zips(p, dest)
        if p.suffix.lower() == ".zip" or zipfile.is_zipfile(p):
            ex = Path(dest) / "extracted" / p.stem
            ex.mkdir(parents=True, exist_ok=True)
            print(f"[ingest] extracting local zip {p} -> {ex}")
            with zipfile.ZipFile(p) as zf:
                _safe_extract(zf, ex)
            return Path(dest) / "extracted"
        sys.exit(f"ERROR: local path '{p}' is neither a directory nor a .zip")

    # 2) http(s) URL
    if src.startswith(("http://", "https://")):
        if "huggingface.co" in src:
            return _hf_download(_hf_repo_from_url(src), dest, revision)
        import urllib.parse

        import httpx
        raw = Path(dest) / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[?#].*$", "", src.split("/")[-1]) or "dataset.zip"
        target = raw / name
        # Only http(s) may be fetched. The branch above already implies this; re-checking the
        # parsed scheme here blocks file:// and custom schemes even if this is called directly.
        scheme = urllib.parse.urlparse(src).scheme.lower()
        if scheme not in ("http", "https"):
            sys.exit(f"ERROR: refusing to download from unsupported URL scheme {scheme!r}: {src}")
        print(f"[ingest] downloading {src} -> {target}")
        # httpx rather than urllib.request.urlretrieve: urlretrieve accepts any scheme the
        # opener supports (file://, ftp://) and writes error pages to disk without complaint.
        # This raises on non-2xx, applies a timeout, and streams so large zips stay off-heap.
        with httpx.stream("GET", src, follow_redirects=True, timeout=60.0) as resp:
            resp.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
        if zipfile.is_zipfile(target):
            ex = Path(dest) / "extracted" / target.stem
            ex.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target) as zf:
                _safe_extract(zf, ex)
            return Path(dest) / "extracted"
        return raw

    # 3) bare HF repo id
    return _hf_download(src, dest, revision)


def _find_split_dir(root, kind, split):
    """Locate the dir holding <kind> for <split>.

    Handles both layouts:
      * SEPARATED:  images/<split>/ + labels/<split>/  (or <split>/images, <split>/labels)
      * CO-LOCATED: <split>/  holds both .jpg and .txt together
    Returns the same co-located dir for either `kind` so _yolo_to_coco can pair
    image-stem -> stem.txt within one directory.
    """
    # Exact-path probes only (each a single stat / dir check) across the split's name aliases
    # and conventional bases. No '*/' globs — those list every depth-1 dir (image folders of
    # 10k+ files) and crawl for minutes over a network (SMB) mount.
    bases = [root, root / "extracted"] + [root / s for s in _CONFIG_SUBDIRS]
    aliases = _split_aliases(split)
    for base in bases:
        for a in aliases:
            for cand in (base / kind / a, base / a / kind):   # separated: images/<split> or <split>/images
                if cand.is_dir():
                    return cand
    for base in bases:
        for a in aliases:
            cand = base / a                                    # co-located: <split>/ holds jpg + txt
            if cand.is_dir() and any(cand.glob("*.jpg")):
                return cand
    return None


def _find_coco_split(root, split):
    """Locate a Roboflow/COCO-export `_annotations.coco.json` for <split>, if any."""
    aliases = {split, {"valid": "val", "validation": "val"}.get(split, split)}
    if split in ("valid", "validation"):
        aliases |= {"valid", "validation", "val"}
    # Probe exact <base>/<alias>/_annotations.coco.json paths (each a single stat). No '*/'
    # globs — those crawl image dirs for minutes over a network mount.
    bases = [root, root / "extracted"] + [root / s for s in _CONFIG_SUBDIRS]
    for base in bases:
        for a in aliases:
            j = base / a / "_annotations.coco.json"
            if j.exists():
                return j
    return None


def _split_aliases(split):
    a = {split}
    if split in ("valid", "validation", "val"):
        a |= {"val", "valid", "validation"}
    if split == "train":
        a |= {"train", "training"}
    if split == "test":
        a |= {"test", "testing"}
    return a


def _is_coco(d):
    return isinstance(d, dict) and "images" in d and "annotations" in d and "categories" in d


def _find_toplevel_coco(root, split):
    """A top-level COCO json named for the split, e.g. train.json / val.json / test.json
    (also annotations_<split>.json, <split>_annotations.json). Returns (path, coco) or None."""
    cands = []
    for a in _split_aliases(split):
        cands += [root / f"{a}.json", root / f"annotations_{a}.json", root / f"{a}_annotations.json",
                  root / f"instances_{a}.json"]
    for c in cands:
        if c.exists():
            try:
                d = json.loads(c.read_text())
            except Exception:
                continue
            if _is_coco(d):
                return c, d
    return None


def _coco_images_root(root, coco):
    """Pick the dir the COCO file_names actually live in (handles flat images/ or JPEGImages/)."""
    fns = [im["file_name"] for im in coco.get("images", [])[:8]]
    for cand in [root / "images", root / "JPEGImages", root / "JPEGimages", root]:
        if cand.is_dir() and fns and all((cand / fn).exists() for fn in fns):
            return cand
    return root / "images" if (root / "images").is_dir() else root


def _load_class_names(root, class_names_arg):
    if class_names_arg:
        names = [l.strip() for l in Path(class_names_arg).read_text().splitlines() if l.strip()]
        if names:
            return names
    # data.yaml: `names: [a, b, ...]` or `names:\n  0: a` — keep it dependency-free.
    # Look ONLY at the root and conventional small config subdirs (exact names) — never '*/'
    # globs, which crawl image dirs for minutes over SMB. Class names also come from COCO
    # `categories` / YOLO ids downstream, so finding nothing here is harmless.
    search_dirs = [root] + [root / s for s in _CONFIG_SUBDIRS]
    yamls, names_files = [], []
    for d in search_dirs:
        if d.is_dir():
            yamls += [d / "data.yaml"] + sorted(d.glob("*.yaml"))
            names_files += sorted(d.glob("*.names"))
    for yml in yamls:
        if not yml.exists():
            continue
        txt = yml.read_text(errors="ignore")
        if "names" in txt:
            try:
                import yaml
                names = yaml.safe_load(txt).get("names")
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names)]
                if names:
                    return list(names)
            except Exception:
                pass
    for nf in names_files:
        names = [l.strip() for l in nf.read_text().splitlines() if l.strip()]
        if names:
            return names
    return None


def _yolo_to_coco(images_dir, labels_dir, class_names, max_images=None):
    """Build a COCO dict from a YOLO split. Reads true image size via PIL."""
    from PIL import Image

    images, annotations = [], []
    ann_id = 1
    img_files = sorted([p for p in images_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if max_images:
        img_files = img_files[:max_images]

    n_classes_seen = 0
    for img_id, img_path in enumerate(img_files):
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": img_path.name,
                       "width": w, "height": h})
        lbl = labels_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            cx, cy, bw, bh = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
            # normalized cx,cy,w,h -> pixel x,y,w,h (top-left)
            pw, ph = bw * w, bh * h
            x, y = cx * w - pw / 2.0, cy * h - ph / 2.0
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": cls,
                                "bbox": [round(x, 2), round(y, 2), round(pw, 2), round(ph, 2)],
                                "area": round(pw * ph, 2), "iscrowd": 0})
            ann_id += 1
            n_classes_seen = max(n_classes_seen, cls + 1)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes_seen)]
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", help="dataset source: HF repo id, HF URL, http(s) .zip URL, "
                                      "local directory, or local .zip")
    ap.add_argument("--extracted-path", help="use an already-extracted YOLO root (skip download)")
    ap.add_argument("--dest", required=True, help="output dir for COCO json + labels.txt")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--class-names", help="text file, one class name per line in id order")
    ap.add_argument("--max-images", type=int, help="cap images per split (smoke runs)")
    ap.add_argument("--revision", default="main",
                    help="Hub revision (branch, tag, or commit SHA). Pin a SHA for reproducible builds.")
    args = ap.parse_args()

    if args.extracted_path:
        root = Path(args.extracted_path)
    elif args.repo_id:
        root = _maybe_download_and_extract(args.repo_id, args.dest, args.revision)
    else:
        sys.exit("ERROR: pass --repo-id or --extracted-path")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    class_names = _load_class_names(root, args.class_names)

    written = {}
    for split in args.splits:
        # (a) top-level COCO json named for the split (train.json / val.json / test.json, etc.),
        # with images in a sibling images/ or JPEGImages/ dir.
        tl = _find_toplevel_coco(root, split)
        if tl is not None:
            coco_json, coco = tl
            images_dir = _coco_images_root(root, coco)
            cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
            if class_names is None and cats:
                class_names = [c["name"] for c in cats]
            out = dest / f"annotations_{split}.json"
            out.write_text(json.dumps(coco))
            written[split] = {"annotations": str(out), "images_dir": str(images_dir),
                              "n_images": len(coco["images"]), "n_boxes": len(coco["annotations"]),
                              "format": "coco"}
            print(f"[ingest] {split}: top-level COCO {coco_json.name} — {len(coco['images'])} imgs, "
                  f"{len(coco['annotations'])} boxes, images at {images_dir} -> {out}")
            continue
        # (b) COCO passthrough (Roboflow export): use the bundled annotations as-is.
        coco_json = _find_coco_split(root, split)
        if coco_json is not None:
            coco = json.loads(coco_json.read_text())
            images_dir = coco_json.parent
            cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
            if class_names is None and cats:
                class_names = [c["name"] for c in cats]
            out = dest / f"annotations_{split}.json"
            out.write_text(json.dumps(coco))
            written[split] = {"annotations": str(out), "images_dir": str(images_dir),
                              "n_images": len(coco["images"]), "n_boxes": len(coco["annotations"]),
                              "format": "coco"}
            print(f"[ingest] {split}: COCO passthrough — {len(coco['images'])} imgs, "
                  f"{len(coco['annotations'])} boxes -> {out}")
            continue
        images_dir = _find_split_dir(root, "images", split)
        labels_dir = _find_split_dir(root, "labels", split)
        if not images_dir or not labels_dir:
            print(f"[ingest] split '{split}' not found (images={images_dir}, labels={labels_dir}) — skipping")
            continue
        coco = _yolo_to_coco(images_dir, labels_dir, class_names, args.max_images)
        # First split with names fixes the class list for all.
        if class_names is None:
            class_names = [c["name"] for c in coco["categories"]]
        out = dest / f"annotations_{split}.json"
        out.write_text(json.dumps(coco))
        # Record the images dir so build_eval_set / tao-finetune-huggingface-model can find files.
        written[split] = {"annotations": str(out), "images_dir": str(images_dir),
                          "n_images": len(coco["images"]), "n_boxes": len(coco["annotations"])}
        print(f"[ingest] {split}: {len(coco['images'])} imgs, {len(coco['annotations'])} boxes -> {out}")

    if not written:
        sys.exit("ERROR: no splits ingested — check the YOLO layout (images/<split>, labels/<split>).")

    (dest / "labels.txt").write_text("\n".join(class_names) + "\n")
    (dest / "ingest_manifest.json").write_text(json.dumps(
        {"root": str(root), "n_classes": len(class_names), "splits": written}, indent=2))
    print(f"[ingest] wrote labels.txt ({len(class_names)} classes) + ingest_manifest.json -> {dest}")


if __name__ == "__main__":
    main()
