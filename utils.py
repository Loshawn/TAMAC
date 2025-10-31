import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


class EpisodicTemporalDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,chunk_size,is_train=True,history_size=49):
        super(EpisodicTemporalDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.is_train=is_train
        self.chunk_size=chunk_size
        self.history_size = history_size  # 每个样本包含多少个时间步的数据（超参数）
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)
    
    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        episode_id = self.episode_ids[index]   
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            future_size=self.history_size+1
            max_start_ts = episode_len
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)  # 选择0到max_start_ts之间的时间步   
                             
            # get observation at start_ts only 获取当前时刻的数据 
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            # 获取当前时刻的图像
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            image_history=[]
            qpos_history=[]
            for i in range(1, self.history_size + 1):
                if start_ts - i >= 0:
                    qpos_history.append(root['/observations/qpos'][start_ts - i])
                    if i % 10 ==0:
                        for cam_name in self.camera_names:
                            image_history.append(root[f'/observations/images/{cam_name}'][start_ts - i])
                else:
                    zero_qpos = np.zeros_like(root['/observations/qpos'][0])
                    qpos_history.append(zero_qpos)  # 使用 append 将全零的 qpos 向量添加到历史数据末尾
                    if i % 10 ==0:
                        for cam_name in self.camera_names:
                            zero_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 修改为适合的图像维度
                            image_history.append(zero_image)  # 使用 append 直接将全零图像添加到历史数据末尾
            all_qpos = [qpos] + qpos_history[:self.history_size]     
            all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names] + image_history
                                                             
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned  

            self.is_sim = is_sim
            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(episode_len)
            is_pad[action_len:] = 1
            if self.is_train:
                image_future = []            
                qpos_future=[]
                for i in range(1, future_size + 1):
                    if start_ts + i < episode_len:
                        # 获取未来时刻的图像
                        if i % 10 ==0:
                            for cam_name in self.camera_names:
                                image_future.append(root[f'/observations/images/{cam_name}'][start_ts + i])
                        qpos_future.append(root['/observations/qpos'][start_ts + i])
                    else:
                        # 如果未来数据不足，用最后一个元素填充
                        if i % 10 ==0:
                            for cam_name in self.camera_names:
                                image_future.append(root[f'/observations/images/{cam_name}'][episode_len - 1])  # 用最后一个时刻的数据填充  
                        qpos_future.append(root['/observations/qpos'][episode_len - 1])  # 用最后一个时刻的 qpos 填充              
                # 将未来图像数据堆叠
                all_cam_images_future = np.stack(image_future, axis=0)      
                all_qpos_future = np.stack(qpos_future, axis=0)          
                # 将当前和未来图像数据连接
                all_cam_images = np.concatenate([all_cam_images, all_cam_images_future], axis=0)         
                all_qpos = np.concatenate([all_qpos, all_qpos_future], axis=0)   
            # 将图像数据堆叠成一个 numpy 数组
            all_cam_images = np.stack(all_cam_images, axis=0)
            all_qpos = np.stack(all_qpos, axis=0)
            # 构造 PyTorch 张量
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(all_qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()
            # 调整图像数据维度
            image_data = torch.einsum('k h w c -> k c h w', image_data)  # 调整图像的通道顺序
            # 归一化图像和标准化动作数据
            image_data = image_data / 255.0  # 图像归一化
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(policy_class, dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size): # !!!!!
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    if policy_class == "ACT":
        train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
        val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    elif policy_class == "TAMAC":
        train_dataset = EpisodicTemporalDataset(train_indices, dataset_dir, camera_names,norm_stats,chunk_size,is_train=True)
        val_dataset = EpisodicTemporalDataset(val_indices, dataset_dir, camera_names, norm_stats,chunk_size,is_train=False)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

def sample_stack_pose():
    # red_box
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    red_box_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    red_box_quat = np.array([1, 0, 0, 0])
    red_box_pose = np.concatenate([red_box_position, red_box_quat])

    # blue_box
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    blue_box_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    blue_box_quat = np.array([1, 0, 0, 0])
    blue_box_pose = np.concatenate([blue_box_position, blue_box_quat])

    return red_box_pose, blue_box_pose

def sample_storage_pose():
    # cube
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    cube_pose = np.concatenate([cube_position, cube_quat])

    # box
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    box_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    box_quat = np.array([1, 0, 0, 0])
    box_pose = np.concatenate([box_position, box_quat])

    return cube_pose, box_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
