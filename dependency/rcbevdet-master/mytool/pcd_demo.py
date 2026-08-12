import open3d as o3d
import numpy as np
# 读取.pcd文件
pcd = o3d.io.read_point_cloud("../data/nuscenes/samples/RADAR_BACK_LEFT/n008-2018-08-01-15-16-36-0400__RADAR_BACK_LEFT__1533151610975167.pcd")

# 输出点云数据的基本信息
print(f"Point cloud has {len(pcd.points)} points.")
print(pcd)
# 获取点云的numpy数组
points = np.asarray(pcd.points)

print(f"The first point: {points[0]}")
print(f"The second point: {points[1]}")
print(f"The 3rd point: {points[2]}")
# 可视化点云
o3d.visualization.draw_geometries([pcd])
