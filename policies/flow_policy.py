"""
FlowPolicy: 3D Flow-based Policy untuk Robot Manipulation
Berdasarkan Consistency Flow Matching dengan 3D vision conditioning
"""
import torch
import torch.nn as nn
from models.point_cloud_encoder import PointCloudEncoder, RGBDPointCloudEncoder
from models.flow_matching import ConsistencyFlowMatching


class FlowPolicy(nn.Module):
    """
    FlowPolicy: Policy generation menggunakan Consistency Flow Matching
    dengan 3D point cloud conditioning
    """
    def __init__(
        self,
        action_dim=9,  # Franka Kitchen: 9-DOF (7 arm + 2 gripper)
        point_cloud_dim=3,  # (x, y, z) coordinates
        feature_dim=256,
        num_points=1024,
        hidden_dim=512,
        use_rgb=False
    ):
        super().__init__()
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.num_points = num_points
        
        # Point cloud encoder
        if use_rgb:
            self.point_cloud_encoder = RGBDPointCloudEncoder(
                feature_dim=feature_dim,
                num_points=num_points
            )
        else:
            self.point_cloud_encoder = PointCloudEncoder(
                point_dim=point_cloud_dim,
                feature_dim=feature_dim,
                num_points=num_points
            )
        
        # Consistency Flow Matching model
        self.flow_matching = ConsistencyFlowMatching(
            action_dim=action_dim,
            condition_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
    def encode_observation(self, point_cloud, rgb_features=None):
        """
        Encode 3D point cloud observation ke feature vector
        
        Args:
            point_cloud: (B, N, 3) point cloud coordinates
            rgb_features: (B, N, 3) RGB features (optional)
        Returns:
            features: (B, feature_dim) encoded features
        """
        if rgb_features is not None:
            return self.point_cloud_encoder(point_cloud, rgb_features)
        else:
            return self.point_cloud_encoder(point_cloud)
    
    def forward(self, point_cloud, t=None, action=None, rgb_features=None, training=True):
        """
        Forward pass untuk training atau inference
        
        Args:
            point_cloud: (B, N, 3) point cloud observation
            t: (B,) time values (untuk training)
            action: (B, action_dim) ground truth action (untuk training)
            rgb_features: (B, N, 3) RGB features (optional)
            training: apakah dalam mode training
        Returns:
            velocity: (B, action_dim) predicted velocity (training)
            action: (B, action_dim) generated action (inference)
        """
        # Encode point cloud
        condition = self.encode_observation(point_cloud, rgb_features)  # (B, feature_dim)
        
        if training and action is not None and t is not None:
            # Training mode: predict velocity
            velocity = self.flow_matching(action, t, condition)
            return velocity
        else:
            # Inference mode: generate action
            action = self.flow_matching.generate(condition, num_steps=1)
            return action
    
    @torch.no_grad()
    def predict(self, point_cloud, rgb_features=None, num_steps=1):
        """
        Predict action dari observation (inference mode)
        
        Args:
            point_cloud: (B, N, 3) point cloud observation
            rgb_features: (B, N, 3) RGB features (optional)
            num_steps: number of inference steps (default: 1 untuk single-step)
        Returns:
            action: (B, action_dim) predicted action
        """
        condition = self.encode_observation(point_cloud, rgb_features)
        action = self.flow_matching.generate(condition, num_steps=num_steps)
        return action
    
    def compute_loss(self, point_cloud, action_gt, t, rgb_features=None):
        """
        Compute training loss untuk FlowPolicy
        Berdasarkan Flow Matching formula untuk straight-line paths
        
        Args:
            point_cloud: (B, N, 3) point cloud observation
            action_gt: (B, action_dim) ground truth action
            t: (B,) time values sampled from [0, 1]
            rgb_features: (B, N, 3) RGB features (optional)
        Returns:
            loss: training loss
        """
        # Encode observation
        condition = self.encode_observation(point_cloud, rgb_features)  # (B, feature_dim)
        
        # Sample noise
        x0 = torch.randn_like(action_gt)  # noise distribution
        x1 = action_gt  # data distribution (ground truth action)
        
        # Straight-line interpolation path: x(t) = (1-t)x_0 + t*x_1
        # Noisy action at time t
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1  # (B, action_dim)
        
        # Target velocity untuk straight-line flow
        # v(x(t), t) = x_1 - x_0 (constant velocity untuk straight line)
        v_target = x1 - x0  # (B, action_dim)
        
        # Predict velocity
        v_pred = self.flow_matching(x_t, t, condition)  # (B, action_dim)
        
        # Flow matching loss: MSE antara predicted dan target velocity
        flow_loss = nn.functional.mse_loss(v_pred, v_target, reduction='mean')
        
        return flow_loss
