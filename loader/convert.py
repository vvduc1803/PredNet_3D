import numpy as np
import open3d as o3d
import pandas as pd

from utils import compute_flow_and_dynamic, SE3_from_frame, create_voxel
import torch
import os

class Create_VoxelMap():
    def __init__(self, root, num_points, vsize, min_xy, max_xy, min_z, max_z):
        self.root = f'{root}/data'
        self.new_root = root
        self.num_points = num_points
        self.vsize = vsize
        self.min_xy = min_xy
        self.max_xy = max_xy
        self.min_z = min_z
        self.max_z = max_z

        self.folders = os.listdir(self.root)

        self.data_path_list = []

        for data_folder in self.folders:
            data_purpose_path = f'{self.root}/{data_folder}/sensor'
            data_purpose = os.listdir(data_purpose_path)[0]

            section_paths = f'{data_purpose_path}/{data_purpose}'
            sections = os.listdir(section_paths)

            for section in sections:
                lidar_paths = f'{section_paths}/{section}/sensors/lidar'
                lidars = sorted(os.listdir(lidar_paths))
                for lidar_id in range(len(lidars)-2):
                    full_curr_lidar_path = f'{lidar_paths}/{lidars[lidar_id]}'
                    full_next_lidar_path = f'{lidar_paths}/{lidars[lidar_id+1]}'

                    self.data_path_list.append([full_curr_lidar_path, full_next_lidar_path])

    def __len__(self):
        return len(self.data_path_list)

    def create_voxel_maps(self):
        data_length = len(self.data_path_list)
        for i, (curr_lidar_path, next_lidar_path) in enumerate(self.data_path_list):

            purpose = curr_lidar_path.split('/')[-5]
            section = curr_lidar_path.split('/')[-4]
            sub_root = os.path.join(*curr_lidar_path.split('/')[0:-3])

            curr_lidar_name = int(curr_lidar_path.split('/')[-1].split('.')[0])
            next_lidar_name = int(next_lidar_path.split('/')[-1].split('.')[0])

            curr_lidar_df = pd.read_feather(curr_lidar_path)

            anntations = pd.read_feather(f'/{sub_root}/annotations.feather')
            city_ego_frame = pd.read_feather(f'/{sub_root}/city_SE3_egovehicle.feather')

            curr_annotations = anntations[anntations['timestamp_ns'] == curr_lidar_name]
            next_annotations = anntations[anntations['timestamp_ns'] == next_lidar_name]

            curr_city_ego_frame = city_ego_frame[city_ego_frame['timestamp_ns'] == curr_lidar_name]
            next_city_ego_frame = city_ego_frame[city_ego_frame['timestamp_ns'] == next_lidar_name]

            curr_lidar, flow, flow_map, status_map = self.create_voxel_map(curr_lidar_df,
                                                                         curr_annotations,
                                                                         next_annotations,
                                                                         curr_city_ego_frame,
                                                                         next_city_ego_frame)


            if not os.path.exists(f'{self.new_root}/data_post_processing/{purpose}/{section}/{curr_lidar_name}_{next_lidar_name}'):
                os.makedirs(f'{self.new_root}/data_post_processing/{purpose}/{section}/{curr_lidar_name}_{next_lidar_name}')

            full_new_root = f'{self.new_root}/data_post_processing/{purpose}/{section}/{curr_lidar_name}_{next_lidar_name}'
            raw_path = f'{full_new_root}/pointcloud_and_flow'
            voxel_path = f'{full_new_root}/occ_data'

            np.savez(raw_path, point_cloud = curr_lidar, scene_flow = flow)
            np.savez(voxel_path, occ_map = status_map, flow_map= flow_map)

            print(f'Extracted: {i+1}/{data_length}')

        print('Finish!')

    def create_voxel_map(self, curr_lidar_df, curr_annotations, next_annotations, curr_city_ego_frame, next_city_ego_frame):
            curr_city_SE3_ego = SE3_from_frame(curr_city_ego_frame)
            next_city_SE3_ego = SE3_from_frame(next_city_ego_frame)

            pc_x_mask = np.logical_and(curr_lidar_df['x'] >= min_xy, curr_lidar_df['x'] <= max_xy - 0.5)
            pc_y_mask = np.logical_and(curr_lidar_df['y'] >= min_xy, curr_lidar_df['y'] <= max_xy - 0.5)
            pc_z_mask = np.logical_and(curr_lidar_df['z'] >= min_z, curr_lidar_df['z'] <= max_z - 0.5)

            mask = pc_x_mask & pc_y_mask & pc_z_mask

            curr_lidar_x = np.array(curr_lidar_df['x']).reshape(-1, 1)
            curr_lidar_y = np.array(curr_lidar_df['y']).reshape(-1, 1)
            curr_lidar_z = np.array(curr_lidar_df['z']).reshape(-1, 1)

            curr_lidar = np.concatenate([curr_lidar_x, curr_lidar_y, curr_lidar_z], axis=1)
            curr_lidar = curr_lidar[mask]

            idxs = np.random.choice(np.sum(mask), num_points)
            curr_lidar = torch.tensor(curr_lidar[idxs], dtype=torch.float32)

            flow, is_dynamic = compute_flow_and_dynamic(curr_lidar,
                                                        curr_annotations,
                                                        next_annotations,
                                                        curr_city_SE3_ego,
                                                        next_city_SE3_ego)

            flow_map, status_map = create_voxel(curr_lidar, flow, is_dynamic, vsize, min_xy, max_xy, min_z, max_z)

            return curr_lidar, flow, flow_map, status_map


if __name__=='__main__':
    num_points = 1000
    vsize = 0.5
    min_xy = -20
    max_xy = -min_xy
    min_z = 0
    max_z = 6
    voxel = Create_VoxelMap('/home/ana/Study/Occupancy_flow/Argoverse_PredNet', num_points, vsize, min_xy, max_xy, min_z, max_z)
    voxel.create_voxel_maps()








