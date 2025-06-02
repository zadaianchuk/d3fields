import os
import sys
sys.path.append(os.getcwd())
import pickle

import cv2
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import torch
import trimesh

from fusion import Fusion, create_init_grid
from utils.draw_utils import aggr_point_cloud_from_data

# Create output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
is_data_from_adamanip = False


# hyper-parameter
t = 50
num_cam = 4

step = 0.004

x_upper = 0.4
x_lower = -0.4
y_upper = 0.3
y_lower = -0.4
z_upper = 0.02
z_lower = -0.2


scene = 'open_window' # 'mug', 'fork', 'shoe'
if scene == 'mug':
    data_path = 'data/2023-09-15-13-21-56-171587' # mug
    pca_path = 'pca_model/mug.pkl'
    query_texts = ['mug']
    query_thresholds = [0.3]
elif scene == 'fork':
    data_path = 'data/2023-09-15-14-15-01-238216' # fork
    pca_path = 'pca_model/fork.pkl'
    query_texts = ['fork']
    query_thresholds = [0.25]
elif scene == 'shoe':
    data_path = 'data/2023-09-11-14-15-50-607452' # shoe
    pca_path = 'pca_model/shoe.pkl'
    query_texts = ['shoe']
    query_thresholds = [0.5]
elif scene == 'open_window':
    data_path = '/ssdstore/azadaia/project_snellius_sync/d3fields/output/adamanip_d3fields/OpenWindow/grasp_env_0'
    pca_path = 'pca_model/mug.pkl'
    query_texts = ['window', "robotic arm"]
    query_thresholds = [0.25]
    is_data_from_adamanip = True
elif scene == 'open_bottle':
    data_path = '/ssdstore/azadaia/project_snellius_sync/AdaManip/d3fields_datasets/rgbd_grasp_OpenBottle_7_eps1_clock0.5_env0/'
    pca_path = 'pca_model/mug.pkl'
    query_texts = ['bottle', "end effector"]
    query_thresholds = [0.25]
    is_data_from_adamanip = True

if is_data_from_adamanip:
    t = 5
    num_cam = 2

    step = 0.01

    x_upper = 1
    x_lower = -1
    y_upper =1
    y_lower = 0
    z_upper = 1
    z_lower = 0.1
    
        
boundaries = {'x_lower': x_lower,
              'x_upper': x_upper,
              'y_lower': y_lower,
              'y_upper': y_upper,
              'z_lower': z_lower,
              'z_upper': z_upper,}

pca = pickle.load(open(pca_path, 'rb'))

fusion = Fusion(num_cam=num_cam, feat_backbone='dinov2', is_data_from_adamanip=is_data_from_adamanip)

colors = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'{t}.png')) for i in range(num_cam)], axis=0) # [N, H, W, C]
depths = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'depth', f'{t}.png'), cv2.IMREAD_ANYDEPTH) for i in range(num_cam)], axis=0) / 1000. # [N, H, W]

H, W = colors.shape[1:3]

def cam_transform(view_matrix):
    
    view_matrix = view_matrix.T
    t = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    view_matrix[:3, :3] = t @ view_matrix[:3, :3]
    view_matrix[:3, 3] = t @ view_matrix[:3, 3]
    # inverse
    R_opencv = view_matrix[:3, :3].T
    t_opencv = -R_opencv @ view_matrix[:3, 3]
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = R_opencv
    camera_pose[:3, 3] = t_opencv
    camera_pose = np.linalg.inv(camera_pose)
    return camera_pose


if is_data_from_adamanip:
    extrinsics = np.stack([(cam_transform(np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy')))) for i in range(num_cam)])
else:
    extrinsics = np.stack([(np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy'))) for i in range(num_cam)])

cam_param = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_params.npy')) for i in range(num_cam)])
print("cam_param: ", cam_param)
intrinsics = np.zeros((num_cam, 3, 3))
intrinsics[:, 0, 0] = cam_param[:, 0]
intrinsics[:, 1, 1] = cam_param[:, 1]
intrinsics[:, 0, 2] = cam_param[:, 2]
intrinsics[:, 1, 2] = cam_param[:, 3]
intrinsics[:, 2, 2] = 1

# assert np.allclose(extrinsics[:, -1], np.array([[0,0,0,1]]))

obs = {
    'color': colors,
    'depth': depths,
    'pose': extrinsics[:, :3], # (N, 3, 4)
    'correct_pose': extrinsics[:, :3], # (N, 3, 4)
    'K': intrinsics,
}

pcd = aggr_point_cloud_from_data(colors[..., ::-1], 
                                 depths, 
                                 intrinsics, 
                                 extrinsics, 
                                 downsample=True, 
                                 boundaries=boundaries, 
                                 is_data_from_adamanip=is_data_from_adamanip)

pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.2)

# save pcd
o3d.io.write_point_cloud(f'{output_dir}/pcd_{scene}.ply', pcd)
print(f'Saved pcd to {output_dir}/pcd_{scene}.ply')



##TODO: fix fusion
##ideas some meshes look weird, maybe because they are not connected

fusion.update(obs)
fusion.text_queries_for_inst_mask_no_track(query_texts, query_thresholds, boundaries=boundaries)

### 3D vis
device = 'cuda'

# visualize mesh
init_grid, grid_shape = create_init_grid(boundaries, step)
init_grid = init_grid.to(device=device, dtype=torch.float32)

print('eval init grid')
with torch.no_grad():
    out = fusion.batch_eval(init_grid, return_names=[])

# extract mesh
print('extract mesh')
vertices, triangles = fusion.extract_mesh(init_grid, out, grid_shape)

# eval mask and feature of vertices
vertices_tensor = torch.from_numpy(vertices).to(device, dtype=torch.float32)
print('eval mesh vertices')
with torch.no_grad():
    out = fusion.batch_eval(vertices_tensor, return_names=['dino_feats', 'mask', 'color_tensor'])

cam = trimesh.scene.Camera(resolution=(1920, 1043), fov=(60, 60))

cam_matrix = np.array([[ 0.87490918, -0.24637599,  0.41693261,  0.63666708],
                       [-0.44229374, -0.75717002,  0.4806972,   0.66457463],
                       [ 0.19725663, -0.60497308, -0.77142556, -1.16125645],
                       [ 0.        , -0.        , -0.        ,  1.        ]])

# create mask mesh
mask_meshes = fusion.create_instance_mask_mesh(vertices, triangles, out)
for i, mask_mesh in enumerate(mask_meshes):
    mask_mesh.export(f'{output_dir}/mask_mesh_{i}_{scene}.ply')
    print(f'Saved mask mesh {i} to {output_dir}/mask_mesh_{i}_{scene}.ply')

# create feature mesh
feature_mesh = fusion.create_descriptor_mesh(vertices, triangles, out, {'pca': pca}, mask_out_bg=True)
feature_mesh.export(f'{output_dir}/feature_mesh_{scene}.ply')
print(f'Saved feature mesh to {output_dir}/feature_mesh_{scene}.ply')

# create color mesh
color_mesh = fusion.create_color_mesh(vertices, triangles, out)
color_mesh.export(f'{output_dir}/color_mesh_{scene}.ply')
print(f'Saved color mesh to {output_dir}/color_mesh_{scene}.ply')

print(f'All meshes saved for scene: {scene}')
