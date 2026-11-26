#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create synchronized camera contact sheets from raw video and canonical tracks."""

import cv2
import json
import math
import hashlib
import argparse

import numpy as np

from pathlib import Path
from evaluate import load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', help='Canonical CSV, never GT for a no-GT review')
    parser.add_argument('--frames', required=True, help='Comma-separated one-based prepared-video frames')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tile-width', type=int, default=480)
    parser.add_argument('--columns', type=int, default=4)
    args = parser.parse_args()
    if args.tile_width < 160 or args.columns < 1:
        parser.error('Use tile-width >= 160 and columns >= 1')
    manifest = Path(args.manifest).resolve()
    data = json.loads(manifest.read_text())
    cameras = data['cameras']
    camera_ids = {str(camera['id']) for camera in cameras}
    if not cameras or len(camera_ids) != len(cameras):
        parser.error('Provide nonempty cameras with unique IDs')
    if data.get('physical_fps') is not None and (not math.isfinite(data['physical_fps']) or data['physical_fps'] <= 0):
        parser.error('Physical FPS must be finite and positive')
    frames = sorted({int(x) for x in args.frames.split(',')})
    if not frames or frames[0] < 1:
        parser.error('Frames must be one-based')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = load_rows(args.predictions) if args.predictions else []
    observations = {}
    for row in rows:
        if row['camera'] not in camera_ids:
            parser.error(f'Unknown prediction camera: {row["camera"]}')
        observations.setdefault((str(row['camera']), row['frame']), []).append(row)
    columns = min(args.columns, len(cameras))
    tile_height = round(args.tile_width * 9 / 16) + 30
    report = []
    captures = []
    try:
        for camera in cameras:
            path = Path(camera['video'])
            path = path if path.is_absolute() else manifest.parent / path
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f'Cannot open {path}')
            captures.append(cap)
        for frame in frames:
            board = np.zeros((math.ceil(len(cameras) / columns) * tile_height,
                              columns * args.tile_width, 3), dtype=np.uint8)
            missing = []
            for index, (camera, cap) in enumerate(zip(cameras, captures)):
                positioned = cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
                ok, picture = cap.read() if positioned else (False, None)
                if ok and abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - frame) > .5:
                    ok = False
                tile = np.zeros((tile_height, args.tile_width, 3), dtype=np.uint8)
                title = f'{camera["id"]} | frame {frame}'
                if data.get('physical_fps'):
                    seconds = float(data.get('source_time_origin_seconds', 0)) + (frame - 1) / data['physical_fps']
                    title += f' | source {seconds:.2f}s'
                if ok:
                    labels = []
                    for row in observations.get((str(camera['id']), frame), []):
                        if any(row[k] is None for k in ('left', 'top', 'width', 'height')):
                            continue
                        color = tuple(60 + b % 180 for b in hashlib.sha256((row['class'] + ':' + row['id']).encode()).digest()[:3])
                        x, y, w, h = (round(row[k]) for k in ('left', 'top', 'width', 'height'))
                        if w > 0 and h > 0:
                            cv2.rectangle(picture, (x, y), (x + w, y + h), color, 2)
                            labels.append((row['class'] + ':' + row['id'], x, y, color))
                    scale = min(args.tile_width / picture.shape[1], (tile_height - 30) / picture.shape[0])
                    resized = cv2.resize(picture, (round(picture.shape[1] * scale), round(picture.shape[0] * scale)))
                    for text, x, y, color in labels:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_width = cv2.getTextSize(text, font, .5, 1)[0][0]
                        font_scale = .5 * min(1, (resized.shape[1] - 8) / max(1, text_width))
                        width, height = cv2.getTextSize(text, font, font_scale, 1)[0]
                        x = min(max(4, round(x * scale)), resized.shape[1] - width - 4)
                        y = min(max(height + 4, round(y * scale) - 4), resized.shape[0] - 4)
                        cv2.rectangle(resized, (x - 2, y - height - 2), (x + width + 2, y + 3), (0, 0, 0), -1)
                        cv2.putText(resized, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)
                    left = (args.tile_width - resized.shape[1]) // 2
                    top = 30 + (tile_height - 30 - resized.shape[0]) // 2
                    tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
                else:
                    title += ' | UNAVAILABLE'
                    missing.append(str(camera['id']))
                cv2.putText(tile, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                y, x = divmod(index, columns)
                board[y * tile_height:(y + 1) * tile_height, x * args.tile_width:(x + 1) * args.tile_width] = tile
            destination = output / f'frame-{frame:06d}.jpg'
            if not cv2.imwrite(str(destination), board):
                raise OSError(f'Could not write {destination}')
            report.append({'frame': frame, 'image': destination.name, 'unavailable_cameras': missing})
    finally:
        for cap in captures:
            cap.release()
    (output / 'frames.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(report)} contact sheets written to {output}. Open the images for visual review.')


if __name__ == '__main__':
    main()
