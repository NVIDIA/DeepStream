#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import numpy as np


def create_frustum_rays(image_size, downsample_factor, dbound):
    """Generate frustum"""
    # make grid in image plane
    ogfH, ogfW = image_size
    fH, fW = ogfH // downsample_factor, ogfW // downsample_factor

    Xs = np.linspace(0, ogfW - 1, fW)
    Ys = np.linspace(0, ogfH - 1, fH)
    Xs, Ys = np.meshgrid(Xs, Ys)
    Zs = np.ones_like(Xs)
    Ws = np.ones_like(Xs)

    # H x W x 4
    rays = np.stack([Xs, Ys, Zs, Ws], axis=-1).astype(np.float32)
    rays_d_bound = [0, 1, dbound[2]]

    # DID
    alpha = 1.5
    d_coords = np.arange(rays_d_bound[2]) / rays_d_bound[2]
    d_coords = np.power(d_coords, alpha)
    d_coords = rays_d_bound[0] + d_coords * (rays_d_bound[1] - rays_d_bound[0])
    d_coords = d_coords[:, np.newaxis, np.newaxis].repeat(fH, axis=1).repeat(fW, axis=2)

    D, _, _ = d_coords.shape

    x_coords = (
        np.linspace(0, ogfW - 1, fW)
        .reshape(1, 1, fW)
        .repeat(D, axis=0)
        .repeat(fH, axis=1)
    )
    y_coords = (
        np.linspace(0, ogfH - 1, fH)
        .reshape(1, fH, 1)
        .repeat(D, axis=0)
        .repeat(fW, axis=2)
    )
    paddings = np.ones_like(d_coords)

    # D x H x W x 3
    frustum = np.stack((x_coords, y_coords, d_coords, paddings), axis=-1)
    return frustum, rays


def get_geometry_rays(
    camera2lidar, intrin_mat, ida_mat, bda_mat, denorms, frustum_rays, rays, dbound
):
    batch_size, num_cams, _, _ = camera2lidar.shape
    ego2sensor_mat = np.linalg.inv(camera2lidar)
    ida_mat_inverse = np.linalg.inv(ida_mat)
    intrin_mat_inverse = np.linalg.inv(intrin_mat)

    H, W = rays.shape[:2]
    B, N = intrin_mat.shape[:2]
    O = (
        np.matmul(
            ego2sensor_mat, np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 1, 4, 1)
        )
    )[..., :3, 0].reshape(B, N, 1, 1, 3, 1)
    n = (
        denorms[:, :3] / np.linalg.norm(denorms[:, :3], axis=-1, keepdims=True)
    ).reshape(B, N, 1, 1, 1, 3)
    P0 = O + dbound[0] * n.reshape(B, N, 1, 1, 3, 1)  # -2
    P1 = O + dbound[1] * n.reshape(B, N, 1, 1, 3, 1)  # 0
    rays_tmp = (
        np.matmul(
            rays.reshape(1, 1, H, W, 4),
            np.matmul(intrin_mat_inverse, ida_mat_inverse)
            .transpose(0, 1, 3, 2)
            .reshape(B, N, 1, 4, 4),
        )
    )[..., :3]
    dirs = np.expand_dims(
        rays_tmp / np.linalg.norm(rays_tmp, axis=-1, keepdims=True), axis=-1
    )

    NPD = np.matmul(n, dirs)
    t0 = np.matmul(n, P0) / NPD
    t1 = np.matmul(n, P1) / NPD

    # B x N x D x H x W x 3
    D, H, W, _ = frustum_rays.shape
    gap = t0 - t1

    points = np.tile(frustum_rays.reshape(1, 1, D, H, W, 4), (B, N, 1, 1, 1, 1))
    points[..., 2] = (
        t0.reshape(B, N, 1, H, W) - points[..., 2] * gap.reshape(B, N, 1, H, W)
    ) * dirs[..., 2, 0].reshape(B, N, 1, H, W)
    ida_tmp = np.transpose(ida_mat_inverse, (0, 1, 3, 2))
    ida_tmp = np.reshape(ida_tmp, (B, N, 1, 1, 4, 4))
    points = np.matmul(points, ida_tmp)
    points[..., :2] *= points[..., [2]]
    matrix = np.matmul(camera2lidar, intrin_mat_inverse)
    if bda_mat is not None:
        matrix = np.matmul(bda_mat[:, np.newaxis], matrix)

    return np.matmul(points, matrix.transpose(0, 1, 3, 2).reshape(B, N, 1, 1, 4, 4))[
        ..., :3
    ].astype(np.float32)


def bev_pool(coords, B, D, H, W, C):
    # X, Y, Z, B
    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )

    indices = np.argsort(ranks)
    coords, ranks = coords[indices], ranks[indices]
    kept = np.ones(coords.shape[0], dtype=bool)
    kept[1:] = ranks[1:] != ranks[:-1]
    interval_starts = np.where(kept)[0].astype(int)
    interval_lengths = np.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = coords.shape[0] - interval_starts[-1]

    ib, ix, iy = coords[kept][:, [3, 0, 1]].T
    interval_output_position = (ib * C * W + ix) * H + iy
    return np.stack(
        [interval_starts, interval_starts + interval_lengths, interval_output_position],
        axis=1,
    ).astype(np.int32), coords[:, 4].astype(np.int32)


def pre_compute(geom_feats, C, bx, dx, nx):
    B, N, D, H, W, _ = geom_feats.shape
    Nprime = B * N * D * H * W

    # flatten indices
    geom_feats = ((geom_feats - (bx - dx / 2.0)) / dx).astype(int)
    geom_feats = geom_feats.reshape(Nprime, 3)
    batch_ix = np.concatenate(
        [np.full((Nprime // B, 1), ix, dtype=int) for ix in range(B)]
    )
    geom_feats = np.concatenate(
        (geom_feats, batch_ix, np.arange(len(batch_ix), dtype=int).reshape(-1, 1)),
        axis=1,
    )

    # filter out points that are outside box
    kept = (
        (geom_feats[:, 0] >= 0)
        & (geom_feats[:, 0] < nx[0])
        & (geom_feats[:, 1] >= 0)
        & (geom_feats[:, 1] < nx[1])
        & (geom_feats[:, 2] >= 0)
        & (geom_feats[:, 2] < nx[2])
    )
    geom_feats = geom_feats[kept]
    intervals, geom_feats = bev_pool(geom_feats, B, nx[2], nx[0], nx[1], C)
    return intervals, geom_feats


def gen_dx_bx(xbound, ybound, zbound):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]], dtype=np.float32)
    bx = np.array(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]], dtype=np.float32
    )
    nx = np.array(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype=np.int64
    )
    return dx, bx, nx


def equation_plane(points):
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = -a * x1 - b * y1 - c * z1
    return np.array([a, b, c, d])


def get_denorm(sweepego2sweepsensor):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate(
        (ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1
    )
    ground_points_cam = np.matmul(sweepego2sweepsensor, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm
