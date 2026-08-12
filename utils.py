"""
utils.py - Data loading and preprocessing helpers for LLM4MG inference
"""

import os

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import uniform_filter
import open3d as o3d
from torchvision import transforms

LIDAR_MAX = 16384   # Keep consistent with dt_gen.py collate_fn's lidar_max
RADAR_MAX = 35      # Keep consistent with dt_gen.py collate_fn's radar_max

IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((320, 480)),
    transforms.ToTensor()
])


def load_rgb(path, transform=None):
    """Load an RGB image and preprocess, returning a [3, H, W] tensor."""
    transform = transform or IMG_TRANSFORMS
    img = Image.open(path)
    return transform(img)


def load_depth(path, transform=None):
    """Load a depth image (convert to grayscale) and preprocess, returning a [1, H, W] tensor."""
    transform = transform or IMG_TRANSFORMS
    depth = Image.open(path)
    if depth.mode != 'L':
        depth = depth.convert('L')
    return transform(depth)


def load_rgbd(rgb_path, depth_path, transform=None):
    """
    Load RGB (+ optional depth) and return a [4, H, W] tensor (R,G,B,D channels).
    If depth is not provided, use the RGB grayscale as the 4th channel.
    """
    transform = transform or IMG_TRANSFORMS
    img = load_rgb(rgb_path, transform)
    if depth_path:
        depth = load_depth(depth_path, transform)
        img = torch.cat((img, depth), dim=0)
    else:
        gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
        img = torch.cat((img, gray), dim=0)
    return img


def load_lidar(path):
    """Load Lidar point cloud (.pcd/.ply/.xyz/.npy/.bin) and return an [N, 3] tensor."""
    if path.endswith('.npy'):
        points = np.load(path)
        return torch.tensor(points, dtype=torch.float32)
    if path.endswith('.bin'):
        raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return torch.tensor(raw[:, :3], dtype=torch.float32)
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    return torch.tensor(points, dtype=torch.float32)


def parse_radar_mat(mat_data):
    """Parse .mat-format radar data into (x, y, z) coordinates. Matches dt_gen.py _parseRadar_."""
    radar_range = np.array(mat_data['radarPoint'][0][0][4])
    if len(radar_range) == 0:
        return np.zeros((0, 3), dtype=float)
    angle = np.array(mat_data['radarPoint'][0][0][-1])[0]
    radar_range = np.concatenate(radar_range).ravel()
    x = radar_range * np.cos(angle)
    y = radar_range * np.sin(angle)
    z = np.zeros_like(radar_range)  # assume Z coordinate is 0
    return np.column_stack((x, y, z))


def extract_radar_xyz(radar_data, cfar_scale=6.0, min_power_db=25):
    """Extract point cloud from 4D radar raw data. Matches dt_gen.py _extract_xyz_."""
    RADAR_PARAMS = {
        'samples': 256, 'chirps': 250, 'rx': 4,
        'adc_sampling': 5e6, 'chirp_slope': 15.015e12,
        'start_freq': 77e9, 'idle_time': 5, 'ramp_end_time': 60
    }
    C = 3e8
    RANGE_RES = ((C * RADAR_PARAMS['adc_sampling']) /
                 (2 * RADAR_PARAMS['samples'] * RADAR_PARAMS['chirp_slope']))
    cube = np.fft.fft(radar_data, axis=1)
    cube = np.fft.fft(cube, axis=2)
    cube = np.fft.fft(cube, n=64, axis=0)
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
    return xyz[~np.all(xyz == 0, axis=1)]


def load_radar(path):
    """Load radar data (.mat or .npy) and return an [N, 3] tensor."""
    if path.endswith('.npy'):
        radar_data = np.load(path)
        # 4D radar raw data goes through CFAR extraction pipeline
        return torch.tensor(extract_radar_xyz(radar_data), dtype=torch.float32)
    mat_data = scipy.io.loadmat(path)
    return torch.tensor(parse_radar_mat(mat_data), dtype=torch.float32)


def pad_points(points, max_num):
    """Pad or sample point clouds to a fixed number. Matches dt_gen.py collate_fn."""
    n = points.size(0)
    if n >= max_num:
        indices = np.random.choice(n, max_num, replace=False)
        return points[indices]
    return F.pad(points, (0, 0, 0, max_num - n), "constant", 0)


def validate_custom_paths(args):
    """Validate required arguments for custom path mode and exit with clear message if missing."""
    missing = []
    for prefix in ['bs', 'vh']:
        for mod in ['rgb', 'lidar', 'radar']:
            p = getattr(args, f'{mod}_{prefix}')
            if p is not None and not os.path.exists(p):
                missing.append(f"--{mod}_{prefix}={p} (file not found)")
    if args.rgb_bs is None and args.rgb_vh is None:
        missing.append("--rgb_bs/--rgb_vh")
    if args.lidar_bs is None and args.lidar_vh is None:
        missing.append("--lidar_bs/--lidar_vh")
    if args.radar_bs is None and args.radar_vh is None:
        missing.append("--radar_bs/--radar_vh")
    if missing:
        raise SystemExit(f"[Custom mode] Missing/invalid arguments:\n  " + "\n  ".join(missing))


def build_custom_batch(args, transform=None):
    """
    Build a single-sample batch (with batch dim) from command-line input paths.

    If only view0 (base station) is provided, view1 (vehicle) will reuse the same data.
    If depth is not provided, use RGB grayscale as the 4th channel.
    """
    transform = transform or IMG_TRANSFORMS

    # If view1 path not provided, reuse view0
    if args.rgb_vh is None and args.rgb_bs is not None:
        args.rgb_vh = args.rgb_bs
    if args.depth_vh is None and args.depth_bs is not None:
        args.depth_vh = args.depth_bs
    if args.lidar_vh is None and args.lidar_bs is not None:
        args.lidar_vh = args.lidar_bs
    if args.radar_vh is None and args.radar_bs is not None:
        args.radar_vh = args.radar_bs

    sample = {}
    for view, prefix in [('view0', 'bs'), ('view1', 'vh')]:
        rgb_path = getattr(args, f'rgb_{prefix}')
        depth_path = getattr(args, f'depth_{prefix}')
        lidar_path = getattr(args, f'lidar_{prefix}')
        radar_path = getattr(args, f'radar_{prefix}')

        if rgb_path is not None:
            sample[f'{view}_RGB'] = load_rgbd(rgb_path, depth_path, transform)
        if lidar_path is not None:
            sample[f'{view}_Lidar'] = load_lidar(lidar_path)
        if radar_path is not None:
            sample[f'{view}_Radar'] = load_radar(radar_path)

    sample['dis'] = args.dis if args.dis is not None else 0.0
    sample['phi'] = args.phi if args.phi is not None else 0.0
    sample['theta'] = args.theta if args.theta is not None else 0.0

    # Assemble batch (add batch dim, pad point clouds; use zero placeholders for missing modalities)
    batch = {}
    for view in ['view0', 'view1']:
        rgb = sample.get(f'{view}_RGB')
        lidar = sample.get(f'{view}_Lidar')
        radar = sample.get(f'{view}_Radar')
        if rgb is not None:
            batch[f'{view}_RGB'] = rgb.unsqueeze(0)
        else:
            # Match dataset __getitem__ empty view: zero-pad [4, H, W]
            batch[f'{view}_RGB'] = torch.zeros(1, 4, 320, 480)
        if lidar is not None:
            batch[f'{view}_Lidar'] = pad_points(lidar, LIDAR_MAX).unsqueeze(0)
        else:
            batch[f'{view}_Lidar'] = torch.zeros(1, LIDAR_MAX, 3)
        if radar is not None:
            batch[f'{view}_Radar'] = pad_points(radar, RADAR_MAX).unsqueeze(0)
        else:
            batch[f'{view}_Radar'] = torch.zeros(1, RADAR_MAX, 3)
    batch['dis'] = torch.tensor([sample['dis']], dtype=torch.float32)
    batch['phi'] = torch.tensor([sample['phi']], dtype=torch.float32)
    batch['theta'] = torch.tensor([sample['theta']], dtype=torch.float32)
    return batch
