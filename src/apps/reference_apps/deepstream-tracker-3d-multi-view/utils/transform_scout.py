# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Optional recomputation of the SCOUT transformed files by bundle adjustment with ground plane correspondences.

This is the procedure that produced the transformed files in assets/scout_transformed. It is disabled
by default and runs only with prepare_scout.py --transform-calib. It needs torch and kornia, which are
imported lazily inside the functions, so importing this module never imports them (install them
with: pip install torch kornia).

The ground plane correspondences come from a json file (undistorted camera points and mesh points).
Collecting them is not part of this module. The results are written as the two json files that
prepare_scout.py loads, in the format of the shipped files.
"""

import json
import importlib.util

import cv2
import numpy as np

from pathlib import Path

if __package__:
    from .prepare_scout import project_points
else:
    from prepare_scout import project_points

TORCH_PACKAGES = ('torch', 'kornia')
INSTALL_HINT = 'pip install torch kornia'
SURROUNDING_METERS = 8
PIXELS_PER_METER = 100
STAIRS_Y = 176
CALIBRATION_FORMAT = 'scout-transformed-calibration-1'
POSITIONS_FORMAT = 'scout-transformed-positions-1'
CALIBRATION_TEXT = ('Transformed camera calibration (undistorted pinhole intrinsics K, rotation R and translation T in the transformed '
                    'map grid frame, world to camera) of the eight SCOUT cameras')
POSITIONS_TEXT = ('Transformed ground plane person positions of SCOUT sequence_1, one entry per (person, frame), recorded once, '
                  'in the transformed map grid frame')
POSITIONS_UNITS = ('x_cm and y_cm are integer centimeters, z_cm is sparse (z_index lists the entries with a nonzero height, '
                   'all others are zero)')
PERSON_TEXT = 'first person id of each frame then differences to the previous id inside the frame'
MAP_FRAME_TEXT = ('similarity transform from the official SCOUT frame to the transformed frame (scale, yaw_deg, translation x and y '
                  'in meters), used only to draw map.png from the official mesh')


def missing_packages():
    """Names of the required packages that are not installed, found without importing them."""
    return [name for name in TORCH_PACKAGES if importlib.util.find_spec(name) is None]


def define_bundle_adjustment():
    """The KorniaBundleAdjustment class, defined on first use because it derives from torch.nn.Module."""
    import torch
    import torch.nn as nn
    from kornia.geometry.camera import PinholeCamera
    from kornia.geometry.conversions import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

    class KorniaBundleAdjustment(nn.Module):
        def __init__(self, params):
            super().__init__()
            self.params = params
            self.init_cameras()
            self.init_3d_positions()
            self.init_correspondences()
            self.extrinsic_calibration_using_correspondences()

        def init_cameras(self):
            device = self.params['device']
            calibrations = self.params['calibrations']
            intrinsics_data = []  # [fx, fy, cx, cy]
            quaternions_data = []
            translations_data = []
            distortions_data = []
            for cam_num in self.params['cam_nums']:
                K, R, T, dist = (calibrations[cam_num][key] for key in ('K', 'R', 'T', 'dist'))
                intrinsics_data.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                quaternions_data.append(rotation_matrix_to_quaternion(torch.tensor(R)).numpy())
                translations_data.append(T.flatten())
                distortions_data.append(dist)
            self.intrinsics = nn.Parameter(torch.tensor(np.array(intrinsics_data), device=device))
            self.quaternions = nn.Parameter(torch.tensor(np.array(quaternions_data), device=device))
            self.translations = nn.Parameter(torch.tensor(np.array(translations_data), device=device))
            self.distortions = torch.tensor(np.array(distortions_data), device=device)
            self.update_cameras()

        def init_3d_positions(self):
            device = self.params['device']
            cam_nums = self.params['cam_nums']
            targets_3d_feet = {}  # (target_id, frame_idx) to [x, y, z] per camera
            targets_visible = {}  # (target_id, frame_idx) to a visibility flag per camera
            targets_2d_feet = {}  # (target_id, frame_idx) to [u, v] per camera, [0, 0] if not visible
            targets_2d_heads = {}  # same for the head
            targets_on_stairs = {}  # (target_id, frame_idx) to a per camera flag, True if world Y is above STAIRS_Y
            print('Initializing annotations')
            for i, cam_num in enumerate(cam_nums):
                for ann in self.params['annotations'][cam_num]:
                    key = (int(ann[1]), int(ann[2]))
                    if key not in targets_3d_feet:
                        targets_3d_feet[key] = np.zeros((len(cam_nums), 3))
                        targets_visible[key] = np.zeros(len(cam_nums), dtype=bool)
                        targets_2d_feet[key] = np.zeros((len(cam_nums), 2))
                        targets_2d_heads[key] = np.zeros((len(cam_nums), 2))
                        targets_on_stairs[key] = np.zeros(len(cam_nums), dtype=bool)
                    targets_3d_feet[key][i] = ann[7:10]
                    targets_visible[key][i] = True
                    targets_2d_feet[key][i] = ann[10:12]
                    targets_2d_heads[key][i] = ann[12:14]
                    targets_on_stairs[key][i] = ann[8] > STAIRS_Y
            pairs = sorted(targets_3d_feet)
            targets, visible, feet, heads, z_trainable_mask = [], [], [], [], []
            print('Computing consensus 3D positions')
            for pair in pairs:
                visible_cams = targets_visible[pair]
                visible_feet = targets_3d_feet[pair][visible_cams]
                loc_mean, loc_std = np.mean(visible_feet, axis=0), np.std(visible_feet, axis=0)
                if np.any(loc_std > 1e-6):
                    raise ValueError(f"{pair} has std > 1e-6: {loc_std}")
                targets.append(loc_mean)
                visible.append(visible_cams)
                feet.append(targets_2d_feet[pair])
                heads.append(targets_2d_heads[pair])
                z_trainable_mask.append(np.any(targets_on_stairs[pair]))
            self.pairs = torch.tensor(np.array(pairs), device=device)  # (N_targets, 2) [person_id, frame_idx]
            self.feet = torch.tensor(np.array(feet), device=device)  # (N_targets, N_cams, 2) [u, v]
            self.heads = torch.tensor(np.array(heads), device=device)  # (N_targets, N_cams, 2) [u, v]
            self.visible = torch.tensor(np.array(visible), device=device)  # (N_targets, N_cams)
            self.z_trainable_mask = torch.tensor(np.array(z_trainable_mask), device=device, dtype=torch.bool)  # (N_targets,)
            self.original_targets = torch.tensor(np.array(targets), device=device, dtype=torch.float32)  # (N_targets, 3) XYZ
            self.targets_xy = nn.Parameter(self.original_targets[:, :2])  # (N_targets, 2)
            self.targets_z = nn.Parameter(self.original_targets[:, 2:3])  # (N_targets, 1)
            on_stairs = self.z_trainable_mask.sum().item()
            print(f'{len(targets)} targets, {on_stairs} on the stairs (trainable Z, world Y > {STAIRS_Y}), '
                  f'{len(targets) - on_stairs} on the ground (Z fixed at 0)')

        def init_correspondences(self):
            device = self.params['device']
            sur, ppm = self.params['surrounding_meters'], self.params['pixels_per_meter']
            correspondences = {}
            for cam_num in self.params['cam_nums']:
                image_points, ground_points = [], []
                for corr in self.params['correspondences'][cam_num]:
                    points_m = (np.array(corr['mesh_point']) - sur * ppm) / ppm
                    points_m[0] = 14 - points_m[0]
                    image_points.append(corr['camera_point'])
                    ground_points.append(np.append(points_m, 0.0))  # z=0, the ground plane
                correspondences[cam_num] = {'img': torch.tensor(np.array(image_points), device=device),
                                            '3D': torch.tensor(np.array(ground_points), device=device)}
            self.correspondences = correspondences

        def extrinsic_calibration_using_correspondences(self):
            device = self.params['device']
            cam_nums = self.params['cam_nums']
            print('Recomputing extrinsics with OpenCV solvePnP')
            for i, cam_num in enumerate(cam_nums):
                points_2d = self.correspondences[cam_num]['img'].detach().cpu().numpy()  # (N, 2)
                points_3d = self.correspondences[cam_num]['3D'].detach().cpu().numpy()  # (N, 3)
                K_3x3 = self.cameras.camera_matrix[i].detach().cpu().numpy()  # (3, 3)
                dist = self.distortions[i].detach().cpu().numpy()
                success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K_3x3, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if not success:
                    raise ValueError(f"solvePnP failed for camera {cam_num}")
                self.quaternions.data[i] = rotation_matrix_to_quaternion(torch.tensor(cv2.Rodrigues(rvec)[0], device=device))
                self.translations.data[i] = torch.tensor(tvec.flatten(), device=device)
                self.update_cameras()
                points_2d_reproj, _ = cv2.projectPoints(points_3d, rvec, tvec, K_3x3, dist)
                reprojection_error = np.linalg.norm(points_2d_reproj.reshape(-1, 2) - points_2d, axis=1).mean()
                print(f"[OpenCV] Camera {cam_num}: {len(points_2d)} correspondences, reprojection error: {reprojection_error:.4f} pixels")

            print('Computing the coordinate system transformation')
            points_old_list = []
            points_new_list = []
            for i, cam_num in enumerate(cam_nums):
                calibration = self.params['calibrations'][cam_num]
                K_old, R_old, t_old, dist_old = calibration['K'], calibration['R'], calibration['T'].flatten(), calibration['dist']
                points_2d = self.correspondences[cam_num]['img'].cpu().numpy()
                points_3d = self.correspondences[cam_num]['3D'].cpu().numpy()
                # Back-project the 2D points to the ground plane z=0 with the old cameras
                for j in range(len(points_2d)):
                    u, v = points_2d[j]
                    pt_dist = np.array([[[u, v]]], dtype=np.float32)
                    u_undist, v_undist = cv2.undistortPoints(pt_dist, K_old, dist_old, P=K_old).flatten()
                    fx, fy, cx, cy = K_old[0, 0], K_old[1, 1], K_old[0, 2], K_old[1, 2]
                    x_norm = (u_undist - cx) / fx
                    y_norm = (v_undist - cy) / fy
                    ray_cam = np.array([x_norm, y_norm, 1.0])
                    ray_world = R_old.T @ ray_cam
                    cam_center = -R_old.T @ t_old
                    if abs(ray_world[2]) > 1e-6:
                        t_param = -cam_center[2] / ray_world[2]
                        if t_param > 0:  # In front of camera
                            points_old_list.append(cam_center + t_param * ray_world)
                            points_new_list.append(points_3d[j])
            points_old = np.array(points_old_list)  # (N_total, 3)
            points_new = np.array(points_new_list)  # (N_total, 3)
            print(f"Collected {len(points_old)} correspondence points for transformation")
            if len(points_old) < 3:
                raise ValueError(f'only {len(points_old)} usable ground plane correspondences, at least 3 are needed '
                                 'to fit the coordinate system transformation')
            # Procrustes/Kabsch for R_transform and t_transform
            points_old_mean = points_old.mean(axis=0)
            points_new_mean = points_new.mean(axis=0)
            points_old_centered = points_old - points_old_mean
            points_new_centered = points_new - points_new_mean
            H_mat = points_old_centered.T @ points_new_centered
            U, _, Vt = np.linalg.svd(H_mat)
            R_transform = Vt.T @ U.T
            if np.linalg.det(R_transform) < 0:  # Ensure a proper rotation
                Vt[-1, :] *= -1
                R_transform = Vt.T @ U.T
            t_transform = points_new_mean - R_transform @ points_old_mean
            print('Transforming targets to the new coordinate system')
            targets_old_3d = self.original_targets.detach().cpu().numpy()  # (N_targets, 3) in meters (OLD system)
            targets_new_3d = (R_transform @ targets_old_3d.T).T + t_transform
            self.targets_xy.data = torch.tensor(targets_new_3d[:, :2], dtype=self.targets_xy.dtype, device=device)
            self.targets_z.data = torch.tensor(targets_new_3d[:, 2:3], dtype=self.targets_z.dtype, device=device)

        def update_cameras(self):
            device = self.params['device']
            count = len(self.params['cam_nums'])
            # 4x4 intrinsics matrices from [fx, fy, cx, cy]
            K_4x4 = torch.eye(4, device=device).unsqueeze(0).expand(count, -1, -1).clone()
            K_4x4[:, 0, 0] = self.intrinsics[:, 0]
            K_4x4[:, 1, 1] = self.intrinsics[:, 1]
            K_4x4[:, 0, 2] = self.intrinsics[:, 2]
            K_4x4[:, 1, 2] = self.intrinsics[:, 3]
            # 4x4 extrinsics matrices from quaternions and translations
            E_4x4 = torch.eye(4, device=device).unsqueeze(0).expand(count, -1, -1).clone()
            E_4x4[:, :3, :3] = quaternion_to_rotation_matrix(self.quaternions)
            E_4x4[:, :3, 3] = self.translations
            height = torch.tensor([self.params['frame_size'][1]] * count, device=device)
            width = torch.tensor([self.params['frame_size'][0]] * count, device=device)
            self.cameras = PinholeCamera(K_4x4, E_4x4, height, width)

        def project_points(self, points_3d):
            points_batch = points_3d.unsqueeze(0).repeat(len(self.params['cam_nums']), 1, 1)
            return self.cameras.project(points_batch)

        def compute_loss(self):
            """Mean squared pixel error of the feet, the heads and the correspondences, all uniformly weighted."""
            device = self.params['device']
            height = self.params['height']
            z_values = torch.zeros(self.targets_xy.shape[0], 1, device=device)
            z_values[self.z_trainable_mask] = self.targets_z[self.z_trainable_mask]
            feet_3d = torch.cat([self.targets_xy, z_values], dim=1)
            heads_3d = torch.cat([self.targets_xy, z_values + height], dim=1)
            feet_proj = self.project_points(feet_3d.float()).transpose(0, 1)
            heads_proj = self.project_points(heads_3d.float()).transpose(0, 1)
            feet_err_list, head_err_list = [], []
            for cam_idx in range(len(self.params['cam_nums'])):
                vis = self.visible[:, cam_idx]
                if not vis.any():
                    continue
                feet_err_list.append((feet_proj[vis, cam_idx] - self.feet[vis, cam_idx].float()) ** 2)
                head_err_list.append((heads_proj[vis, cam_idx] - self.heads[vis, cam_idx].float()) ** 2)
            feet_errors = torch.cat(feet_err_list, dim=0) if feet_err_list else torch.zeros(0, 2, device=device)
            head_errors = torch.cat(head_err_list, dim=0) if head_err_list else torch.zeros(0, 2, device=device)
            corr_err_list = []
            for i, cam_num in enumerate(self.params['cam_nums']):
                corr_obs = self.correspondences[cam_num]['img'].float()
                corr_3d = self.correspondences[cam_num]['3D'].float()
                corr_err_list.append((self.project_points(corr_3d)[i] - corr_obs) ** 2)
            corr_errors = torch.cat(corr_err_list, dim=0) if corr_err_list else torch.zeros(0, 2, device=device)
            total_loss = feet_errors.mean() + head_errors.mean() + corr_errors.mean()
            return feet_errors, head_errors, corr_errors, total_loss

    return KorniaBundleAdjustment


def load_correspondences(path, cameras):
    """Load the ground plane correspondences, converting mesh points when the grid parameters differ.

    Returns a dictionary from camera number to a list of dictionaries with camera_point and mesh_point.
    """
    with Path(path).open() as stream:
        loaded = json.load(stream)
    old_surrounding = loaded.pop('surrounding_meters', None)
    old_per_meter = loaded.pop('pixels_per_meter', None)
    correspondences = {int(key): value for key, value in loaded.items() if key.isdigit()}
    absent = [camera for camera in cameras if not correspondences.get(camera)]
    if absent:
        raise ValueError(f'{path}: no correspondences for cameras {absent}')
    if None not in (old_surrounding, old_per_meter) and (old_surrounding, old_per_meter) != (SURROUNDING_METERS, PIXELS_PER_METER):
        for pairs in correspondences.values():
            for pair in pairs:
                meters = (np.asarray(pair['mesh_point'], dtype=float) - old_surrounding*old_per_meter)/old_per_meter
                pair['mesh_point'] = (meters*PIXELS_PER_METER + SURROUNDING_METERS*PIXELS_PER_METER).tolist()
    return correspondences


def build_params(official, undistorted_matrices, annotations, correspondences, height, frame_size, max_iter, learning_rate,
                 save_interval, transformed_out, region, names, device):
    """The parameter dictionary of the ported bundle adjustment.

    official maps a camera to its official calibration (K, R, T, dist). annotations maps a camera to
    the official rows [frame, person, x, y, w, h, X, Y, Z] of all frames. The boxes do not take part
    in the adjustment and are not shifted here. names are the two output file names, calibration first.
    """
    cams = sorted(official)
    calibrations, rows = {}, {}
    for cam in cams:
        calib = {'K': undistorted_matrices[cam], 'R': official[cam]['R'], 'T': official[cam]['T'].reshape(3, 1),
                 'dist': np.zeros_like(official[cam]['dist'])}
        calibrations[cam] = calib
        official_rows = annotations[cam]
        array = np.column_stack([np.full(len(official_rows), cam), official_rows[:, 1], official_rows[:, 0], official_rows[:, 2:6],
                                 official_rows[:, 6:9]])
        head = array[:, 7:10].copy()
        head[:, 2] += height
        rows[cam] = np.hstack([array, *(project_points(points, calib['K'], calib['R'], calib['T'], calib['dist'])
                                        for points in (array[:, 7:10], head))])
    return {'cam_nums': cams, 'calibrations': calibrations, 'annotations': rows, 'correspondences': correspondences,
            'surrounding_meters': SURROUNDING_METERS, 'pixels_per_meter': PIXELS_PER_METER, 'height': height,
            'frame_size': frame_size, 'max_iter': max_iter, 'learning_rate': learning_rate, 'save_interval': save_interval,
            'transformed_out': str(transformed_out), 'region': region, 'names': names, 'device': device, 'log_every': 100}


def encode_positions(persons, frames, x_cm, y_cm, z_cm):
    """Columnar positions as stored in assets/scout_transformed (integer centimeters, sparse z)."""
    persons, frames = np.asarray(persons, dtype=np.int64), np.asarray(frames, dtype=np.int64)
    order = np.lexsort((persons, frames))
    persons, frames = persons[order], frames[order]
    x, y, z = (np.asarray(values, dtype=np.int64)[order] for values in (x_cm, y_cm, z_cm))
    unique, counts = np.unique(frames, return_counts=True)
    deltas = persons.copy()
    start = 0
    for count in counts:
        deltas[start + 1:start + count] = np.diff(persons[start:start + count])
        start += count
    nonzero = np.nonzero(z)[0]
    return {'format': POSITIONS_FORMAT, 'description': POSITIONS_TEXT, 'units': POSITIONS_UNITS, 'count': int(len(persons)),
            'frames': unique.tolist(), 'per_frame': counts.tolist(), 'person_delta': PERSON_TEXT, 'person': deltas.tolist(),
            'x_cm': x.tolist(), 'y_cm': y.tolist(), 'z_index': nonzero.tolist(), 'z_cm': z[nonzero].tolist()}


def fit_similarity(source, target):
    """Least squares similarity (scale, yaw in degrees, translation) mapping 2D source points to target points."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    a, b = source - source_mean, target - target_mean
    u, s, vt = np.linalg.svd(a.T @ b/len(a))
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = vt.T @ np.diag([1.0, sign]) @ u.T
    scale = (s*np.array([1.0, sign])).sum()/(a**2).sum()*len(a)
    return float(scale), float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))), target_mean - scale*rotation @ source_mean


def format_centimeters(values):
    """Integer centimeters of the two decimal rounded values, the precision the ground truth uses."""
    return [int(round(float('%.2f' % value)*100)) for value in values]


def write_recorded_files(directory, cameras, persons, frames, xy, z, original_xy, losses, region, names):
    """Write the transformed calibration and positions json files (names, calibration first), the format prepare_scout.py loads.

    region (xmin, xmax, ymin, ymax) selects the positions that anchor the map frame.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inside = (xy[:, 0] > region[0]) & (xy[:, 0] < region[1]) & (xy[:, 1] > region[2]) & (xy[:, 1] < region[3])
    scale, yaw, shift = fit_similarity(original_xy[inside], xy[inside]) if inside.sum() > 2 else (1.0, 0.0, np.zeros(2))
    calibration = {'format': CALIBRATION_FORMAT, 'description': CALIBRATION_TEXT, 'cameras': cameras,
                   'map_frame': {'description': MAP_FRAME_TEXT, 'scale': round(scale, 12), 'yaw_deg': round(yaw, 12),
                                 'translation': [round(float(shift[0]), 12), round(float(shift[1]), 12)]},
                   'provenance': {'method': 'bundle_adjustment_kornia_quaternions', 'persons': int(len(np.unique(persons))),
                                  'person_frame_pairs': int(len(persons)), 'loss_metrics': losses}}
    with (directory/names[0]).open('w') as stream:
        json.dump(calibration, stream, indent=1)
    positions = encode_positions(persons, frames, format_centimeters(xy[:, 0]), format_centimeters(xy[:, 1]), format_centimeters(z))
    with (directory/names[1]).open('w') as stream:
        stream.write(json.dumps(positions, separators=(',', ':')))
    return directory/names[0], directory/names[1]


def save_checkpoint(params, state):
    """Write the best state so far as the transformed json files (the research converter wrote per frame annotation files)."""
    from kornia.geometry.conversions import quaternion_to_rotation_matrix
    loss = state['losses']['total']
    print(f'Saving checkpoint (loss {loss:.4f})')
    cameras = {}
    for i, cam_num in enumerate(params['cam_nums']):
        K_param = state['intrinsics'][i].cpu().numpy()
        K = np.array([[K_param[0], 0, K_param[2]], [0, K_param[1], K_param[3]], [0, 0, 1]])
        R = quaternion_to_rotation_matrix(state['quaternions'][i:i + 1])[0].cpu().numpy()
        t = state['translations'][i].cpu().numpy().reshape(3)
        cameras[str(cam_num)] = {'K': K.tolist(), 'R': R.tolist(), 'T': [float(value) for value in t]}
    pairs = state['pairs'].cpu().numpy()
    xy = state['targets_xy'].cpu().numpy().astype(np.float64)
    z_mask = state['z_trainable_mask'].cpu().numpy()
    z = np.where(z_mask, state['targets_z'].cpu().numpy().reshape(-1), 0.0).astype(np.float64)
    original_xy = state['original_targets'][:, :2].cpu().numpy().astype(np.float64)
    original_z = state['original_targets'][:, 2].cpu().numpy()
    losses = {'total_loss': float(loss), 'feet_rmse_pixels': float(state['losses']['feet_rmse']),
              'head_rmse_pixels': float(state['losses']['head_rmse']),
              'correspondences_rmse_pixels': float(state['losses']['corr_rmse'])}
    paths = write_recorded_files(params['transformed_out'], cameras, pairs[:, 0], pairs[:, 1], xy, z, original_xy, losses,
                                 params['region'], params['names'])
    print(f"  Total: {loss:.3f} | Feet: {losses['feet_rmse_pixels']:.3f}px | Head: {losses['head_rmse_pixels']:.3f}px | "
          f"Corr: {losses['correspondences_rmse_pixels']:.3f}px | mean Z change {np.abs(z - original_z).mean():.3f}m")
    print(f"  Saved {paths[0]} and {paths[1]}")
    return paths


def run_bundle_adjustment(params):
    """Adam on the bundle adjustment model, saving the best state at the save interval and at the end."""
    import torch
    import torch.optim as optim
    model = define_bundle_adjustment()(params)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-5)
    best_loss = float('inf')
    best_model_state = None
    first_loss = None
    print(f"Bundle adjustment: {params['max_iter']} iterations, learning rate {params['learning_rate']}, "
          f"best state saved every {params['save_interval']} iterations")
    for i in range(params['max_iter']):
        feet_errors, head_errors, corr_errors, loss = model.compute_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        model.update_cameras()
        with torch.no_grad():
            current_loss = loss.item()
            feet_rmse = torch.sqrt(feet_errors.mean()).item()
            head_rmse = torch.sqrt(head_errors.mean()).item()
            corr_rmse = torch.sqrt(corr_errors.mean()).item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = {
                    'intrinsics': model.intrinsics.data.clone(), 'quaternions': model.quaternions.data.clone(),
                    'translations': model.translations.data.clone(), 'original_targets': model.original_targets.clone(),
                    'targets_xy': model.targets_xy.data.clone(), 'targets_z': model.targets_z.data.clone(),
                    'z_trainable_mask': model.z_trainable_mask, 'pairs': model.pairs,
                    'losses': {'total': current_loss, 'feet_rmse': feet_rmse, 'head_rmse': head_rmse, 'corr_rmse': corr_rmse}}
            if i > 0 and i % params['save_interval'] == 0 and best_model_state is not None:
                save_checkpoint(params, best_model_state)
            if first_loss is None:
                first_loss = current_loss
            if i % params['log_every'] == 0 or i == params['max_iter'] - 1:
                print(f"Iteration {i}: loss: {current_loss:.4f}, best: {best_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.2e}, "
                      f"feet: {feet_rmse:.4f}px, head: {head_rmse:.4f}px, corr: {corr_rmse:.4f}px", flush=True)
    if best_model_state is None:
        raise ValueError('the bundle adjustment never reached a finite loss, check the correspondences and the annotations')
    print(f'Bundle adjustment done, best loss {best_loss:.4f}, final loss {current_loss:.4f}')
    paths = save_checkpoint(params, best_model_state)
    return {'paths': paths, 'best_loss': best_loss, 'first_loss': first_loss, 'final_loss': current_loss}


def transform_calibration(official, undistorted_matrices, annotations, correspondences, transformed_out, region, names, height,
                          frame_size, max_iter, learning_rate=1e-3, save_interval=1000, device=None):
    """Run the bundle adjustment and write the two transformed json files (names, calibration first) into transformed_out."""
    missing = missing_packages()
    if missing:
        raise ImportError(f'Missing {", ".join(missing)}. Install with: {INSTALL_HINT}')
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    params = build_params(official, undistorted_matrices, annotations, correspondences, height, frame_size, max_iter,
                          learning_rate, save_interval, transformed_out, region, names, device)
    return run_bundle_adjustment(params)
