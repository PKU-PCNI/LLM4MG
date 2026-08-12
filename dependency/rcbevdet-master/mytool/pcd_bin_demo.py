import numpy as np
import open3d as o3d


def load_bin_to_pcd(bin_file_path):
    # 使用numpy加载二进制文件，每个点占用4个float32 (x, y, z, intensity)
    point_cloud_data = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)

    # 提取 x, y, z (忽略强度)
    points = point_cloud_data[:, :3]

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# 加载 .pcd.bin 文件
bin_file = "../data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin"
pcd = load_bin_to_pcd(bin_file)

# 输出点云的基本信息
print(f"Point cloud has {len(pcd.points)} points.")

# 可视化点云
o3d.visualization.draw_geometries([pcd])
