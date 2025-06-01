import numpy as np
import torch

def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    depth_bar = 2.5
    mask = np.logical_and(mask, depth <  depth_bar)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))

    fx = proj_matrix[0, 0] * width / 2.0
    fy = proj_matrix[1, 1] * height / 2.0
    # fx = 2 / proj_matrix[0, 0]
    # fy = 2 / proj_matrix[1, 1]
    # Extract principal point
    cx = (proj_matrix[0, 2] * width + width) / 2.0
    cy = (proj_matrix[1, 2] * height + height) / 2.0

    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = - (pos_x - cx) * depth[mask]* 2 / (proj_matrix[0, 0] * width)
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] * 2 / (proj_matrix[1, 1] * height)
    fgpcd[:, 2] = depth[mask]
    pcd = fgpcd
    poses = extrinsics # (N, 4, 4)
    pose = np.linalg.inv(pose)
    
    trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0) # (4, N)
    trans_pcd = trans_pcd[:3, :].T # (N, 3)
    return trans_pcd


def depth_image_to_point_cloud_GPU( camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar):
        # time1 = time.time()
        depth_buffer = camera_tensor
        # Get the camera view matrix and invert it to transform points from camera to world space
        vinv = camera_view_matrix_inv
        print(f"vinv: {vinv}")
        # Get the camera projection matrix and get the necessary scaling
        # coefficients for deprojection
        proj = camera_proj_matrix
        fu = 2 / proj[0, 0]
        fv = 2 / proj[1, 1]
        centerU = width / 2
        centerV = height / 2
        # print(f"depth_buffer: {depth_buffer.min()}, {depth_buffer.max()}, depth_bar: {depth_bar}")
        # print(f"fu: {fu}")
        # print(f"fv: {fv}")
        # print(f"centerU: {centerU}")
        # print(f"centerV: {centerV}")
        Z = depth_buffer
        X = -(u - centerU) * (2 * Z) / (width * proj[0, 0])
        Y = (v - centerV) * (2 * Z) / (height * proj[1, 1])
        Z = Z.view(-1)
        valid = Z > -depth_bar
        X = X.view(-1)
        Y = Y.view(-1)
        position = torch.vstack((X, Y, Z, torch.ones(len(X))))[:, valid]
        position = position.permute(1, 0) # (n, 4)
        position = position @ vinv # (n, 4)
        points = position[:, 0:3] # (n, 3)
        return points