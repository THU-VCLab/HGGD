import numpy as np
import open3d

from .configs import config

# from open3d.open3d.geometry import voxel_down_sample, estimate_normals, orient_normals_towards_camera_location


class PointCloud:

    def __init__(self, cloud: open3d.geometry.PointCloud, visualization=False):
        self.cloud = cloud
        self.visualization = visualization
        self.normals = None if len(np.asarray(
            cloud.normals)) <= 0 else np.asarray(cloud.normals)

    def remove_outliers(self):
        num_points_threshold = config.NUM_POINTS_THRESHOLD
        radius_threshold = config.RADIUS_THRESHOLD
        self.cloud.remove_radius_outlier(nb_points=num_points_threshold,
                                         radius=radius_threshold)
        if self.visualization:
            open3d.visualization.draw_geometries([self.cloud])

    def voxelize(self, voxel_size=config.VOXEL_SIZE):
        self.cloud.voxel_down_sample(voxel_size=voxel_size)
        # self.cloud = voxel_down_sample(self.cloud, voxel_size=voxel_size)
        if self.visualization:
            open3d.visualization.draw_geometries([self.cloud])

    def estimate_normals(self, camera_pos=np.zeros(3)):
        normal_radius = config.NORMAL_RADIUS
        normal_max_nn = config.NORMAL_MAX_NN
        self.cloud.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn),
            fast_normal_computation=False)
        # estimate_normals(self.cloud,
        #     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))#,
        # #    fast_normal_computation=False)
        self.cloud.normalize_normals()
        if True:
            # orient_normals_towards_camera_location(self.cloud, camera_pos)
            self.cloud.orient_normals_towards_camera_location(camera_pos)
        if self.visualization:
            open3d.visualization.draw_geometries([self.cloud])

        self.normals = np.asarray(self.cloud.normals)
