"""
Wrapper untuk Franka Kitchen environment dengan vision support
Mengkonversi RGB-D observations menjadi point clouds
"""
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from typing import Dict, Tuple, Optional
import open3d as o3d


class PointCloudRenderer:
    """Helper class untuk convert RGB-D images ke point clouds"""
    
    @staticmethod
    def depth_to_pointcloud(depth_image, rgb_image, camera_intrinsic, camera_extrinsic=None):
        """
        Convert depth image ke point cloud
        
        Args:
            depth_image: (H, W) depth values
            rgb_image: (H, W, 3) RGB values
            camera_intrinsic: camera intrinsic matrix (3x3)
            camera_extrinsic: camera extrinsic matrix (4x4) (optional)
        Returns:
            point_cloud: (N, 3) point coordinates
            rgb_points: (N, 3) RGB values per point
        """
        H, W = depth_image.shape
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        
        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Convert ke 3D points
        z = depth_image.flatten()
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy
        
        # Filter invalid depth values
        valid = z > 0
        points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        rgb_points = rgb_image.reshape(-1, 3)[valid] / 255.0  # Normalize to [0, 1]
        
        # Apply extrinsic transformation if provided
        if camera_extrinsic is not None:
            ones = np.ones((points.shape[0], 1))
            points_homo = np.concatenate([points, ones], axis=-1)
            points_transformed = (camera_extrinsic @ points_homo.T).T
            points = points_transformed[:, :3]
        
        return points, rgb_points
    
    @staticmethod
    def downsample_pointcloud(points, rgb_points, num_points=1024):
        """
        Downsample point cloud ke jumlah tertentu
        
        Args:
            points: (N, 3) point coordinates
            rgb_points: (N, 3) RGB values
            num_points: target number of points
        Returns:
            points_downsampled: (num_points, 3)
            rgb_downsampled: (num_points, 3)
        """
        if points.shape[0] <= num_points:
            # Pad dengan zeros jika kurang
            pad_size = num_points - points.shape[0]
            points = np.concatenate([points, np.zeros((pad_size, 3))], axis=0)
            rgb_points = np.concatenate([rgb_points, np.zeros((pad_size, 3))], axis=0)
            return points, rgb_points
        
        # Downsample menggunakan farthest point sampling atau random sampling
        indices = np.random.choice(points.shape[0], num_points, replace=False)
        return points[indices], rgb_points[indices]


class FrankaKitchenVisionWrapper(gym.Wrapper):
    """
    Wrapper untuk Franka Kitchen environment dengan point cloud observations
    """
    def __init__(
        self,
        env,
        num_points=1024,
        camera_width=84,
        camera_height=84,
        use_rgb=True,
        camera_name='agentview'
    ):
        super().__init__(env)
        self.num_points = num_points
        self.use_rgb = use_rgb
        self.camera_name = camera_name
        
        # Camera intrinsic (default values, bisa disesuaikan)
        self.camera_intrinsic = np.array([
            [camera_width * 2, 0, camera_width / 2],
            [0, camera_height * 2, camera_height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Render point cloud
        self.point_cloud_renderer = PointCloudRenderer()
        
        # Update observation space
        if use_rgb:
            self.observation_space = gym.spaces.Dict({
                'point_cloud': gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(num_points, 3), dtype=np.float32
                ),
                'rgb_features': gym.spaces.Box(
                    low=0, high=1,
                    shape=(num_points, 3), dtype=np.float32
                ),
                'state': env.observation_space
            })
        else:
            self.observation_space = gym.spaces.Dict({
                'point_cloud': gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(num_points, 3), dtype=np.float32
                ),
                'state': env.observation_space
            })
    
    def _render_pointcloud(self):
        """
        Render point cloud dari environment
        Menggunakan depth dan RGB dari camera
        
        Note: Ini adalah placeholder implementation.
        Dalam implementasi nyata, gunakan MuJoCo renderer untuk mendapatkan
        depth dan RGB images, lalu convert ke point cloud.
        """
        # Placeholder: Generate synthetic point cloud
        # TODO: Implementasi actual rendering dari MuJoCo camera untuk point cloud
        # Dalam implementasi nyata, gunakan:
        # 1. env.render() untuk mendapatkan RGB image
        # 2. MuJoCo depth rendering untuk mendapatkan depth image
        # 3. Convert RGB-D ke point cloud menggunakan PointCloudRenderer
        
        num_points = self.num_points
        
        # Generate synthetic point cloud (placeholder)
        # Distribusi point cloud sekitar origin dengan beberapa variasi
        points = np.random.randn(num_points, 3) * 0.5  # (N, 3) point coordinates
        points[:, 2] = np.abs(points[:, 2]) + 0.5  # Ensure z > 0 (depth values)
        
        rgb_points = None
        if self.use_rgb:
            # Generate synthetic RGB features (placeholder)
            rgb_points = np.random.rand(num_points, 3)  # (N, 3) RGB values [0, 1]
        
        return points.astype(np.float32), rgb_points.astype(np.float32) if rgb_points is not None else None
    
    def reset(self, **kwargs):
        """Reset environment dan return point cloud observation"""
        obs, info = self.env.reset(**kwargs)
        
        # Render point cloud
        point_cloud, rgb_features = self._render_pointcloud()
        
        # Construct new observation
        new_obs = {
            'point_cloud': point_cloud,
            'state': obs
        }
        if self.use_rgb and rgb_features is not None:
            new_obs['rgb_features'] = rgb_features
        
        return new_obs, info
    
    def step(self, action):
        """Step environment dan return point cloud observation"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Render point cloud
        point_cloud, rgb_features = self._render_pointcloud()
        
        # Construct new observation
        new_obs = {
            'point_cloud': point_cloud,
            'state': obs
        }
        if self.use_rgb and rgb_features is not None:
            new_obs['rgb_features'] = rgb_features
        
        return new_obs, reward, terminated, truncated, info


def make_franka_kitchen_env(
    env_name='FrankaKitchen-v1',
    tasks_to_complete=['microwave', 'kettle'],
    num_points=1024,
    use_rgb=True,
    **kwargs
):
    """
    Factory function untuk membuat Franka Kitchen environment dengan vision wrapper
    
    Args:
        env_name: name of the environment
        tasks_to_complete: list of tasks
        num_points: number of points in point cloud
        use_rgb: whether to use RGB features
    Returns:
        env: wrapped environment
    """
    env = gym.make(env_name, tasks_to_complete=tasks_to_complete, **kwargs)
    env = FrankaKitchenVisionWrapper(
        env,
        num_points=num_points,
        use_rgb=use_rgb
    )
    return env
