import os
import sys

import torch
import numpy as np
import natsort
from PIL import Image
import scipy.io

import open3d as o3d
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import uniform_filter

class SynthSoMV2I(Dataset):
    """
    Custom dataset for loading multi-view multi-sensor data (RGB, Lidar, Radar, Depth) for training.

    This class is responsible for loading and processing data from various sources like RGB images, depth maps,
    lidar point clouds, and radar data. It supports multiple frequencies (60GHz, sub6, etc.) and
    applies data transformations for data augmentation.

    Noted that Depth data will be loaded in image tensor (4-channel (RGB-D) image).
    """

    def __init__(self, data_path, data_categories:list,
                 view_list:list, img_transform=None,
                 transform2=None, fc="60GHz",
                 ):
        super(SynthSoMV2I, self).__init__()
        self.data_categories = data_categories
        self.transform = img_transform if img_transform is not None else transforms.ToTensor()
        self.view_list = view_list
        self.transform2 = transform2
        self.fc = fc
        self.ca_dict = {'RGB':'camera_data', 'Depth':'depth_data/png_output',
                        'Lidar':'lidar_data','Radar':"radar_data/radarPoint2"}

        self.data_dict = {}
        view_idx = 0

        print("Loading data from {} ...".format(data_path))
        print("Selected sensing data: {} ".format(self.data_categories))
        print("Using sensoring data: {}".format(view_list))

        for view in view_list:
            self.data_dict[f'view{view_idx}'] = {}
            if view == '-':
                continue
            num = 0
            for ca in data_categories:
                if os.name == "nt":
                    path = os.path.join(data_path, view, self.ca_dict[ca]) + "\\"
                elif sys.platform.startswith("linux"):
                    path = os.path.join(data_path, view, self.ca_dict[ca]) + "/"
                else:
                    raise ValueError("Unsupported operating system")

                file_list = natsort.natsorted(os.listdir(path))
                file_path = list(map(lambda f:path+f, file_list))
                self.data_dict[f'view{view_idx}'][ca] = file_path
                if num == 0: num = len(file_path)
                else: assert num == len(file_path), f"The num of {ca} in {view} is not correct"
            view_idx += 1

        if len(self.data_dict) > 1  and self.fc != 'measurement':
            total_num = 0
            first_num = 0
            for view in self.data_dict.keys():
                if first_num == 0: first_num = len(self.data_dict[view][data_categories[0]])
                total_num += len(self.data_dict[view][data_categories[0]])
            assert first_num == total_num/len(self.data_dict), f"The num of {view} is not correct"

        self.label = {}
        if fc == "60GHz":
            print(f"Using fc {fc}")
            matfile = scipy.io.loadmat(os.path.join(data_path, 'Label', '60_lowVTD_urban.mat' ))
            antenna_path = os.path.join(data_path, 'urban60GHz_rsf_veh_dis_angle.txt')
            self.dis = scipy.io.loadmat(os.path.join(data_path, 'dis', 'dis_urban60GHz.mat'))['dis'][0]
            self.dis = torch.from_numpy(self.dis).float()
        else:
            raise ValueError(f"{fc} must in ['60GHz']")

        self.phi = []
        self.theta = []
        with open(antenna_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    phi = float(parts[2])
                    theta = float(parts[3])
                    self.phi.append(phi)
                    self.theta.append(theta)
        self.phi = torch.tensor(np.array(self.phi))
        self.theta = torch.tensor(np.array(self.theta))
        self.label['LoS_c'] = torch.tensor(np.array(matfile['NLoS_component']).squeeze())
        self.label['power_max6'] = torch.tensor(np.vstack(matfile['power_max6_normalize'].squeeze()), dtype=torch.float32)
        self.label['tau_max6'] = torch.tensor(np.vstack(matfile['tau_max6_train'].squeeze()), dtype=torch.float32)

        print("Successfully load {} Snapshot.".format(len(self.data_dict['view0']['RGB'])))

    def __len__(self):
        """
        Return the length of the dataset (number of samples).
        """
        return len(self.data_dict['view0']['RGB'])

    def _parseRadar_(self, mat_data):
        """
        Parse radar data from MATLAB format to 3D coordinates (x, y, z)
        Args:
            mat_data (dict): Radar data loaded from a .mat file
        Returns:
            np.ndarray: A 2D array of 3D coordinates (x, y, z).
        """
        radar_range = np.array(mat_data['radarPoint'][0][0][4])
        if len(radar_range) == 0:
            return np.zeros((0, 3), dtype=float)


        angle = np.array(mat_data['radarPoint'][0][0][-1])[0]

        radar_range = np.concatenate(radar_range).ravel()
        x = radar_range * np.cos(angle)
        y = radar_range * np.sin(angle)
        z = np.zeros_like(radar_range)  # assume Z coordinate is 0
        xyz = np.column_stack((x, y, z))
        return xyz

    def _extract_xyz_(self, radar_data: np.ndarray,
                      cfar_scale: float = 6.0,
                      min_power_db: float = 25
                      ) -> np.ndarray:
        """
        Extract 3D coordinates (x, y, z) from 4D radar data tensor.

        Args:
            radar_data (np.ndarray): 4D radar data array with shape (rx, samples, chirps).
            cfar_scale (float): Scale factor for CA-CFAR background subtraction (default is 6.0).
            min_power_db (float): Minimum power threshold in dB for point inclusion (default is 25).

        Returns:
            np.ndarray: 2D array of extracted 3D points (x, y, z).
        """

        # ---------- PARAMS ----------
        RADAR_PARAMS = {
            'samples': 256,
            'chirps': 250,
            'rx': 4,
            'adc_sampling': 5e6,
            'chirp_slope': 15.015e12,
            'start_freq': 77e9,
            'idle_time': 5,
            'ramp_end_time': 60
        }
        C = 3e8
        RANGE_RES = ((C * RADAR_PARAMS['adc_sampling']) /
                     (2 * RADAR_PARAMS['samples'] * RADAR_PARAMS['chirp_slope']))

        # ---------- 3D FFT ----------
        cube = np.fft.fft(radar_data, axis=1)  # range
        cube = np.fft.fft(cube, axis=2)  # Doppler
        cube = np.fft.fft(cube, n=64, axis=0)  # azimuth
        cube = np.fft.fftshift(cube, axes=(0, 2))
        power = np.abs(cube) ** 2
        power_db = 10 * np.log10(power + 1e-12)

        mask = power_db > min_power_db

        background = uniform_filter(power, size=(5, 5, 5), mode='constant')
        mask &= power > (cfar_scale * background)

        rx_idx, r_idx, _ = np.where(mask)
        if len(r_idx) == 0:
            return np.empty((0, 3))

        n_az = 64
        az_vec = np.linspace(-np.pi / 3, np.pi / 3, n_az)
        range_vec = np.arange(RADAR_PARAMS['samples']) * RANGE_RES

        r = range_vec[r_idx]
        theta = az_vec[rx_idx]

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(x)

        xyz = np.column_stack((x, y, z))
        xyz = xyz[~np.all(xyz == 0, axis=1)]
        return xyz

    def __getitem__(self, idx):
        outputs = {}
        view_idx = 0
        for view in self.data_dict.keys():
            if len(self.data_dict[view]) == 0:

                outputs[f'view{view_idx}_RGB'] = torch.zeros(4,320,480)
                outputs[f'view{view_idx}_Lidar'] = torch.zeros(16384, 3)
                outputs[f'view{view_idx}_Radar'] = torch.zeros(35, 3)
                view_idx += 1
                continue
            for ca in self.data_categories:
                if ca == 'Depth': continue
                elif ca == 'RGB':
                    img = Image.open(self.data_dict[view][ca][idx])
                    img = self.transform(img)
                    if 'Depth' in self.data_categories:
                        if self.fc == 'measurement' and '-' in self.view_list:
                            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                            gray = gray.unsqueeze(0)
                            img = torch.cat((img, gray), dim=0)
                        else:
                            depth = Image.open(self.data_dict[view]['Depth'][idx])
                            if depth.mode != 'L':
                                depth = depth.convert('L')
                            depth = self.transform(depth)
                            img = torch.cat((img, depth), dim=0)
                            if self.transform2 is not None:
                                img = self.transform2(img)
                        outputs[f'view{view_idx}_'+ca] = img
                elif ca == 'Lidar':
                    pcd = o3d.io.read_point_cloud(self.data_dict[view][ca][idx])
                    points = np.array(pcd.points)
                    points_tensor = torch.tensor(points, dtype=torch.float32)
                    outputs[f'view{view_idx}_' + ca] = points_tensor
                elif ca == 'Radar':
                    if self.fc == "measurement" and self.data_dict[view][ca][idx][-3:] == "npy":
                        radar_data = np.load(self.data_dict[view][ca][idx])
                        xyz = self._extract_xyz_(radar_data)
                    else:
                        mat_data = scipy.io.loadmat(self.data_dict[view][ca][idx])
                        xyz = self._parseRadar_(mat_data)
                    # If no points were extracted, keep empty array (no debug print)
                    outputs[f'view{view_idx}_'+ca] = torch.tensor(xyz, dtype=torch.float32)

            view_idx += 1

        outputs['dis'] = self.dis[idx]
        outputs['phi'] = self.phi[idx]
        outputs['theta'] = self.theta[idx]
        outputs['LoS_c'] = self.label['LoS_c'][idx]
        outputs['power_max6'] = self.label['power_max6'][idx]
        outputs['tau_max6'] = self.label['tau_max6'][idx]

        return outputs



    def collate_fn(self, batch):
        """
        Collate function for DataLoader to handle batches of variable-size data
        Args:
            batch (list): A list of data samples (dictionaries)
        Returns:
            dict: A batch of data with tensors stacked along the batch dimension.
        """
        keys = list(batch[0].keys())
        lidar_max = 16384
        radar_max= 35
        outputs = {}
        ca_nums = len(self.data_categories)-1 if 'Depth' in self.data_categories else len(self.data_categories)
        for i in range( (len(keys)-6) // ca_nums ):
            if 'RGB' in self.data_categories:
                outputs[f'view{i}_RGB'] = torch.stack([item[f'view{i}_RGB'] for item in batch])
            if 'Lidar' in self.data_categories:
                lidar_batch = [item[f'view{i}_Lidar'] for item in batch]
                lidar_padded = []
                for points in lidar_batch:
                    n = points.size(0)
                    if n >= lidar_max:
                        indices = np.random.choice(len(points), lidar_max, replace=False)
                        padded = points[indices]
                    else:
                        padded = F.pad(points, (0, 0, 0, lidar_max - n), "constant", 0)
                    lidar_padded.append(padded)
                lidar_padded = torch.stack(lidar_padded, dim=0)
                outputs[f'view{i}_Lidar'] = lidar_padded
            if 'Radar' in self.data_categories:
                radar_batch = [item[f'view{i}_Radar'] for item in batch]
                radar_padded = []
                for points in radar_batch:
                    n = points.size(0)
                    if n >= radar_max:
                        indices = np.random.choice(len(points), radar_max, replace=False)
                        padded = points[indices]
                    else:
                        padded = F.pad(points, (0, 0, 0, radar_max - n), "constant", 0)
                    radar_padded.append(padded)
                radar_padded = torch.stack(radar_padded, dim=0)
                outputs[f'view{i}_Radar'] = radar_padded


        outputs['dis'] = torch.stack([item[f'dis'] for item in batch])
        outputs['phi'] = torch.stack([item[f'phi'] for item in batch])
        outputs['theta'] = torch.stack([item[f'theta'] for item in batch])

        # Lable
        outputs['LoS_c'] = torch.stack([item[f'LoS_c'] for item in batch])
        outputs['power_max6'] = torch.stack([item[f'power_max6'] for item in batch])
        outputs['tau_max6'] = torch.stack([item[f'tau_max6'] for item in batch])
        return outputs
