import pandas as pd
import numpy as np
import torch
from kornia.geometry.linalg import transform_points
from scipy.spatial.transform import Rotation

import pandas as pd
import numpy as np
from typing import Final

from torch.serialization import location_tag
from torch.utils.hipify.hipify_python import value

LIDAR_COLUMNS: Final = ("x", "y", "z", "intensity")
QWXYZ_COLUMNS: Final = ("qw", "qx", "qy", "qz")
TRANSLATION_COLUMNS: Final = ("tx_m", "ty_m", "tz_m")
XYZLWH_QWXYZ_COLUMNS: Final = (
    TRANSLATION_COLUMNS + ("length_m", "width_m", "height_m") + QWXYZ_COLUMNS
)

import torch
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.linalg import transform_points

from av2api.src.av2.torch.structures.cuboids import Cuboids
from av2api.src.av2.torch.structures.flow import cuboids_to_id_cuboid_map

from av2api.src.av2.evaluation.scene_flow.constants import (
    CATEGORY_TO_INDEX,
    SCENE_FLOW_DYNAMIC_THRESHOLD,
)


from mayavi import mlab

SCENE_FLOW_DYNAMIC_THRESHOLD = 0.05
BOUNDING_BOX_EXPANSION = 0.2
def vertices_m(arr, rot, trans):
    r"""Return the cuboid vertices in the destination reference frame.

        5------4
        |\\    |\\
        | \\   | \\
        6--\\--7  \\
        \\  \\  \\ \\
    l    \\  1-------0    h
     e    \\ ||   \\ ||   e
      n    \\||    \\||   i
       g    \\2------3    g
        t      width.     h
         h.               t.

    Returns:
        (8,3) array of cuboid vertices.
    """
    unit_vertices_obj_xyz_m = np.array(
        [
            [+1, +1, +1],  # 0
            [+1, -1, +1],  # 1
            [+1, -1, -1],  # 2
            [+1, +1, -1],  # 3
            [-1, +1, +1],  # 4
            [-1, -1, +1],  # 5
            [-1, -1, -1],  # 6
            [-1, +1, -1],  # 7
        ],
    )
    dims_lwh_m = np.repeat(arr, 8, axis=0)

    dims_lwh_m = dims_lwh_m.reshape(-1, 8, 3)

    # Transform unit polygons.

    vertices_obj_xyz_m = (dims_lwh_m / 2.0) * unit_vertices_obj_xyz_m
    # vertices_dst_xyz_m = self.dst_SE3_object.transform_point_cloud(
    #     vertices_obj_xyz_m
    # )

    # Finally, return the polygons.
    # return vertices_obj_xyz_m
    matrixes = []
    # print(vertices_obj_xyz_m.shape)
    # print(rot.shape)
    # print(trans.shape)
    # input()
    # print(trans.shape)
    for matrix, rotate, tran in zip(vertices_obj_xyz_m, rot, trans):
        a = matrix @ rotate.T + tran
        matrixes.append(a)
    matrixes = np.array(matrixes)
    return matrixes

def inside_test(points , cube3d):
    """
    cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
    points = array of points with shape (N, 3).

    Returns the indices of the points array which are outside the cube3d
    """
    b1,b2,b3,b4,t1,t2,t3,t4 = cube3d

    dir1 = (t1-b1)
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = (b2-b1)
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = (b4-b1)
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t3)/2.0

    dir_vec = points - cube3d_center

    # print(np.absolute(np.dot(dir_vec, dir1)) * 2 <= size1)

    res1 = np.absolute(np.dot(dir_vec, dir1)) * 2 <= size1
    res2 = np.absolute(np.dot(dir_vec, dir2)) * 2 <= size2
    res3 = np.absolute(np.dot(dir_vec, dir3)) * 2 <= size3
    mask = res1 & res2 & res3

    return mask

def SE3_from_frame(frame: pd.DataFrame) -> Se3:
    """Build SE(3) object from `pandas` DataFrame.

    Notation:
        N: Number of rigid transformations.

    Args:
        frame: (N,4) Pandas DataFrame containing quaternion coefficients.

    Returns:
        SE(3) object representing a (N,4,4) tensor of homogeneous transformations.
    """
    quaternion_npy = frame[list(QWXYZ_COLUMNS)].to_numpy().astype(float)
    quat_wxyz = Quaternion(torch.as_tensor(quaternion_npy, dtype=torch.float32)[None])
    rotation = So3(quat_wxyz)

    translation_npy = (
        frame[list(TRANSLATION_COLUMNS)].to_numpy().astype(np.float32)
    )
    translation = torch.as_tensor(translation_npy, dtype=torch.float32)[None]
    dst_SE3_src = Se3(rotation, translation)
    dst_SE3_src.rotation._q.requires_grad_(False)
    dst_SE3_src.translation.requires_grad_(False)
    return dst_SE3_src

def compute_flow_and_dynamic(lidar1, curr_annotations_frame, next_annotations_frame, city_SE3_ego1, city_SE3_ego2):
    is_valid = torch.ones(len(lidar1), dtype=torch.bool)
    category_inds = torch.zeros(len(lidar1), dtype=torch.uint8)

    ego1_SE3_ego0 = city_SE3_ego2.inverse() * city_SE3_ego1


    rigid_flow = (
        (transform_points(ego1_SE3_ego0.matrix(), lidar1[:, :3][None])[0] - lidar1)
        .float()
        .detach()
    )

    flow = rigid_flow.clone()
    check_dynamic = torch.zeros_like(flow)


    cuboids1 = Cuboids(curr_annotations_frame)
    current_cuboid_map = cuboids_to_id_cuboid_map(cuboids1)

    cuboids2 = Cuboids(next_annotations_frame)
    next_cuboid_map = cuboids_to_id_cuboid_map(cuboids2)

    for i, id in enumerate(current_cuboid_map):
        c0 = current_cuboid_map[id]
        c0.length_m += BOUNDING_BOX_EXPANSION  # the bounding boxes are a little too tight sometimes
        c0.width_m += BOUNDING_BOX_EXPANSION
        obj_pts_npy, obj_mask_npy = c0.compute_interior_points(lidar1.numpy())
        obj_pts, obj_mask = torch.as_tensor(
            obj_pts_npy, dtype=torch.float32
        ), torch.as_tensor(obj_mask_npy)
        category_inds[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]

        if id in next_cuboid_map:

            c1 = next_cuboid_map[id]
            c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
            flow[obj_mask] = (
                    torch.as_tensor(
                        c1_SE3_c0.transform_point_cloud(obj_pts_npy),
                        dtype=torch.float32,
                    )
                    - obj_pts
            )

            check_dynamic[obj_mask] = (
                    torch.as_tensor(
                        c1_SE3_c0.transform_point_cloud(obj_pts_npy),
                        dtype=torch.float32,
                    )
                    - obj_pts
            )
        else:
            is_valid[obj_mask] = 0

    dynamic_norm = torch.linalg.vector_norm(check_dynamic, dim=-1)
    is_dynamic = dynamic_norm >= SCENE_FLOW_DYNAMIC_THRESHOLD

    return flow, is_dynamic

def create_voxel(curr_lidar, flow, is_dynamic, vsize, min_xy, max_xy, min_z, max_z):

    xy_shape = int(max_xy * 2 / vsize)
    z_shape = int(max_z * 2 / vsize)

    status_map = np.zeros((3, xy_shape, xy_shape, z_shape))
    status_map[1] = np.ones((xy_shape, xy_shape, z_shape))

    flow_map = np.zeros((3, xy_shape, xy_shape, z_shape))

    curr_lidar[:, 0] = curr_lidar[:, 0] - min_xy
    curr_lidar[:, 1] = curr_lidar[:, 1] - min_xy
    curr_lidar[:, 2] = curr_lidar[:, 2] - min_z

    for location, flow_value, status_value in zip(curr_lidar, flow, is_dynamic):
        location = location / vsize
        location_x, location_y, location_z = int(location[0]), int(location[1]), int(location[2])
        flow_map[:, location_x, location_y, location_z] = flow_value
        status_map[0, location_x, location_y, location_z] = 1
        status_map[1, location_x, location_y, location_z] = 0
        if status_value:
            status_map[2, location_x, location_y, location_z] = 1

    return flow_map, status_map