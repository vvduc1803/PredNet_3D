import torch
import torch.utils.data as data
import os
import pandas as pd
import numpy as np

from utils import compute_flow_and_dynamic, SE3_from_frame


class MapData(data.Dataset):
    def __init__(self, root, nt, num_points=10000, mode='train', sequence_start_mode='all'):
        self.root = root
        self.nt = nt
        self.num_points = num_points
        self.mode = mode
        self.sequence_start_mode = sequence_start_mode
        self.data_items = []

        if mode == 'train':
            self.data_folders = [i for i in os.listdir(self.root) if i[0] == 't']
        elif mode == 'val':
            self.data_folders = [str(i).startswith('val') for i in os.listdir(self.root)]
        elif mode == 'test':
            self.data_folders = [str(i).startswith('test') for i in os.listdir(self.root)]

        for data_folder in self.data_folders:
            lidar_path = f'{self.root}/{data_folder}/sensors/lidar'
            lidar_ls = sorted(os.listdir(lidar_path))
            lidar_ls = [f'{data_folder}/'+i for i in lidar_ls]
            lidar_splits = [lidar_ls[i: i+self.nt+1] for i in range(0, len(lidar_ls)-self.nt-1)]

            self.data_items.extend(lidar_splits)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        lidars = []
        flows = []
        is_dynamics = []

        lidar_paths = self.data_items[idx]
        data_folder, _ = lidar_paths[0].split('/')
        anntations = pd.read_feather(f'{self.root}/{data_folder}/annotations.feather')
        city_ego_frame = pd.read_feather(f'{self.root}/{data_folder}/city_SE3_egovehicle.feather')

        idxs = np.random.choice(95000, self.num_points)

        for lidar_id in range(len(lidar_paths)-1):
            curr_lidar_file = lidar_paths[lidar_id].split('/')[1]
            next_lidar_file = lidar_paths[lidar_id+1].split('/')[1]
            curr_lidar_name = int(curr_lidar_file.split('.')[0])
            next_lidar_name = int(next_lidar_file.split('.')[0])
            curr_lidar_full_path = f'{self.root}/{data_folder}/sensors/lidar/{curr_lidar_file}'
            next_lidar_full_path = f'{self.root}/{data_folder}/sensors/lidar/{next_lidar_file}'

            curr_lidar_df = pd.read_feather(curr_lidar_full_path)

            curr_annotations = anntations[anntations['timestamp_ns']==curr_lidar_name]
            next_annotations = anntations[anntations['timestamp_ns']==next_lidar_name]

            curr_city_ego_frame = city_ego_frame[city_ego_frame['timestamp_ns'] == curr_lidar_name]
            next_city_ego_frame = city_ego_frame[city_ego_frame['timestamp_ns'] == next_lidar_name]

            curr_city_SE3_ego = SE3_from_frame(curr_city_ego_frame)
            next_city_SE3_ego = SE3_from_frame(next_city_ego_frame)

            curr_lidar_x = np.array(curr_lidar_df['x']).reshape(-1, 1)
            curr_lidar_y = np.array(curr_lidar_df['y']).reshape(-1, 1)
            curr_lidar_z = np.array(curr_lidar_df['z']).reshape(-1, 1)

            curr_lidar = np.concatenate([curr_lidar_x, curr_lidar_y, curr_lidar_z], axis=1)


            curr_lidar = torch.tensor(curr_lidar[idxs], dtype=torch.float32)
            # curr_lidar = torch.tensor(curr_lidar, dtype=torch.float32)

            flow, is_dynamic = compute_flow_and_dynamic(curr_lidar,
                                                        curr_annotations,
                                                        next_annotations,
                                                        curr_city_SE3_ego,
                                                        next_city_SE3_ego)

            lidars.append(curr_lidar)
            flows.append(flow)
            is_dynamics.append(is_dynamic)

        lidars = torch.stack(lidars, dim=0)
        flows = torch.stack(flows, dim=0)
        is_dynamics = torch.stack(is_dynamics, dim=0)

        return lidars, flows, is_dynamics

if __name__=='__main__':
    root = '/home/ana/Study/Occupancy_flow/Argoverse_PredNet/data'
    data = MapData(root, 5, 10000)
    lidars, flows, is_dynamics = data.__getitem__(0)
    # print(len(data))

    print(lidars[0])
    print(lidars[1])
    print(flows[0])