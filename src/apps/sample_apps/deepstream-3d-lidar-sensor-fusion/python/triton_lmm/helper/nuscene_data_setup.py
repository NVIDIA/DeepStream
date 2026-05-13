# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import pathlib
import shutil
import subprocess

import numpy as np
from nuscenes.nuscenes import NuScenes

CAM_SENSORS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
LIDAR_SENSOR = "LIDAR_TOP"
NUSCENE_ROOT = "dataset/nuscenes"
DS3D_FUSION_WORKSPACE = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion"
# NUSCENE_V10_MINI_URL = "https://motional-nuscenes.s3-ap-northeast-1.amazonaws.com/public/v1.0/v1.0-mini.tgz"
NUSCENE_V10_MINI_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
NUSCENE_V10_MINI_TAR = "v1.0-mini.tgz"


def printlog(msg, **kvargs):
    print(msg, **kvargs)    # Print statement to be skipped # noqa: T201


def check_download_nuscene_mini(data_dir) -> bool:
    """ downloading nuscene v1.0-mini dataset if it does not exists
    """
    folder = pathlib.Path(data_dir)
    testfile = folder / "v1.0-mini/sample_data.json"
    printlog(f"checking {str(testfile)}")

    if testfile.exists():
        printlog(f"file exists, skip downloading {NUSCENE_V10_MINI_TAR}")
        return True
    printlog(f"file does not exist, start downloading {NUSCENE_V10_MINI_URL}")
    dst = folder / NUSCENE_V10_MINI_TAR
    folder.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["curl", "-o", str(dst), NUSCENE_V10_MINI_URL], check=True)
        printlog(f"mini dataset downloaded successfully to {str(dst)}")
    except subprocess.CalledProcessError as e:
        printlog(f"Error downloading file: {e}")
        return False

    try:
        printlog(f"uncompressing dataset {str(dst)}")
        subprocess.run(["tar", "-pxvf", str(dst), "-C", str(folder)], check=True)
        printlog("uncompressing done")
    except subprocess.CalledProcessError as e:
        printlog(f"Error uncompressing {str(dst)}: {e}")
        return False
    return True


def _file_copy(src, dst):
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def f_data(data):
    return np.array2string(data, formatter={'float': lambda x: f'{x:.4f},'})


class DataSceneDS3D:
    """ setup sample data for ds3d(deepstream-3d) sensor fusion tests """

    def __init__(self, dataroot: str, version: str = "v1.0-mini", scene_idx=0):
        self._src_root = dataroot
        self._nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        assert self._nusc
        self._scene = self._nusc.scene[scene_idx]
        assert self._scene
        printlog(
            f"scene: {self._scene['token']}, name: {self._scene['name']}, has {self._scene['nbr_samples']} samples")

    def copy_to_ds3d_workspace(self, todir: str, scene_idx=0):
        """copy scene<idx> to ds3d sensor fusion workspace

        Args:
            scene_idx (int): scene index number
        """
        assert self._nusc
        assert self._scene
        first_sample_token = self._scene["first_sample_token"]
        # last_sample_token = self._scene["last_sample_token"]

        cur_token = first_sample_token
        sample_num = min(self._scene["nbr_samples"], 100)
        for i in range(sample_num):
            sample = self._nusc.get("sample", cur_token)
            self._sample_data_copy(sample, todir, i)
            cur_token = sample["next"]

    def print_calibration(self):
        """Print lidar, camera intrinsic, extrinsic calibration data
        """
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        first_sample_token = self._scene["first_sample_token"]
        sample = self._nusc.get("sample", first_sample_token)
        data = sample["data"]
        lidar_sensor = LIDAR_SENSOR
        cam_sensors = CAM_SENSORS
        lidar_data = self._nusc.get("sample_data", data[lidar_sensor])
        printlog(f"lidar-sensor token: {lidar_data['calibrated_sensor_token']}")
        lidar_calib = self._nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        lidar_to_ego = transform_matrix(lidar_calib["translation"], Quaternion(lidar_calib["rotation"]), inverse=False)
        for cam in cam_sensors:
            cam_data = self._nusc.get("sample_data", data[cam])
            cam_calib = self._nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            ego_to_cam = transform_matrix(cam_calib["translation"], Quaternion(cam_calib["rotation"]), inverse=True)
            lidar_to_cam = np.dot(ego_to_cam, lidar_to_ego)
            cam_to_lidar = np.linalg.inv(lidar_to_cam)
            cam_intrinsic = np.asarray(cam_calib["camera_intrinsic"])
            cam_intrinsic_4x4 = np.eye(4)
            cam_intrinsic_4x4[:3, :3] = cam_intrinsic
            lidar2image = np.matmul(cam_intrinsic_4x4, lidar_to_cam)
            printlog(f"{cam} intrinsic: {f_data(cam_intrinsic_4x4)}")
            printlog(f"lidar_to_{cam} extrinsic: {f_data(lidar_to_cam)}")
            printlog(f"{cam}_to_lidar extrinsic: {f_data(cam_to_lidar)}")
            printlog(f"lidar_to_{cam}_image matrix: {f_data(lidar2image)}")

    def _sample_data_copy(self, sample, todir: str, idx=0):
        """copy nusene lidar and camera dataset into ds3d fusion workspace
        the final folder director look like
        """
        dataset = sample["data"]
        data_folder = pathlib.Path(todir)
        lidar_set = {LIDAR_SENSOR}
        sensors = [LIDAR_SENSOR] + CAM_SENSORS
        assert self._nusc

        for sensor in sensors:
            sample_data = self._nusc.get("sample_data", dataset[sensor])
            sensor_folder = data_folder / sensor
            dst_path = f"{sensor_folder}/{idx:0>6}-{sensor}"
            if sensor in lidar_set:    # is lidar
                dst_path += ".bin"
            else:
                dst_path += ".jpg"
            src_path = self._src_root + "/" + sample_data["filename"]
            printlog(f"copying file {src_path} to {dst_path}")
            _file_copy(src_path, dst_path)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=NUSCENE_ROOT,
        help="nuscene dataset top directory, (default: %(default)s)",
    )

    parser.add_argument(
        "--ds3d_fusion_workspace",
        type=str,
        default=DS3D_FUSION_WORKSPACE,
        help="ds3d(deepstream-3d) sensor fusion worksapce, (default: %(default)s)",
    )

    parser.add_argument(
        "--disable_data_copy",
        action="store_true",
        help="disable nuscene data copy to ds3d(deepstream-3d) sensor fusion worksapce",
    )

    parser.add_argument(
        "--print_calibration",
        action="store_true",
        help="Enable printing nuscene calibration data",
    )

    return parser


def main():
    """example:
        nuscene_data_setup.py --data_dir=dataset/nuscene \
            --ds3d_fusion_workspace=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion \
            --print_calibration
        After data copy, ds3d_fusion_workspace directory tree would look like
        data
        ├── nuscene
        │    ├── CAM_BACK
        │    │    ├── 000000-CAM_BACK.jpg
        │    ├── CAM_BACK_LEFT
        │    │    ├── 000000-CAM_BACK_LEFT.jpg
        │    ├── CAM_BACK_RIGHT
        │    │    ├── 000000-CAM_BACK_RIGHT.jpg
        │    ├── CAM_FRONT
        │    │    ├── 000000-CAM_FRONT.jpg
        │    ├── CAM_FRONT_LEFT
        │    │    ├── 000000-CAM_FRONT_LEFT.jpg
        │    ├── CAM_FRONT_RIGHT
        │    │    ├── 000000-CAM_FRONT_RIGHT.jpg
        │    └── LIDAR_TOP
        │        ├── 000000-LIDAR_TOP.bin
    """
    parser = setup_parser()
    args = parser.parse_args()
    if not check_download_nuscene_mini(args.data_dir):
        printlog(f"nuscene data check failed. try manually download {NUSCENE_V10_MINI_URL}")

    ds3d_fusion_data_dir = args.ds3d_fusion_workspace + "/data/nuscene"
    scene_idx = 0
    scene = DataSceneDS3D(dataroot=args.data_dir, version="v1.0-mini", scene_idx=scene_idx)
    assert scene

    if not args.disable_data_copy:
        printlog(f"start copying data to folder {ds3d_fusion_data_dir}")
        scene.copy_to_ds3d_workspace(ds3d_fusion_data_dir)
        printlog(f"copying data to folder {ds3d_fusion_data_dir} is done")
    if args.print_calibration:
        printlog("printing calibration data")
        scene.print_calibration()


if __name__ == "__main__":
    """Main cli."""
    main()
