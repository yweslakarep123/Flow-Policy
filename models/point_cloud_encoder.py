"""
Point Cloud Encoder untuk 3D vision representation
Berdasarkan PointNet++ architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PointNetLayer(nn.Module):
    """PointNet layer dengan MLP"""
    def __init__(self, in_dim, out_dim, bn=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim) if bn else nn.Identity(),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.mlp(x)


class PointCloudEncoder(nn.Module):
    """
    Point Cloud Encoder untuk encoding 3D point clouds
    Architecture berbasis PointNet++
    """
    def __init__(self, point_dim=3, feature_dim=256, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # Input encoding
        self.input_mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Hierarchical feature extraction
        self.layer1 = PointNetLayer(64, 128)
        self.layer2 = PointNetLayer(128, 256)
        self.layer3 = PointNetLayer(256, 512)
        
        # Global feature aggregation dengan max pooling
        self.global_pool = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3) tensor of point coordinates
        Returns:
            features: (B, feature_dim) global point cloud features
        """
        B, N, D = point_cloud.shape
        
        # Flatten untuk MLP processing
        x = rearrange(point_cloud, 'b n d -> (b n) d')
        
        # Input encoding
        x = self.input_mlp(x)
        x = rearrange(x, '(b n) d -> b n d', b=B, n=N)
        
        # Hierarchical feature extraction dengan max pooling
        x1 = self.layer1(rearrange(x, 'b n d -> (b n) d'))
        x1 = rearrange(x1, '(b n) d -> b n d', b=B, n=N)
        f1 = torch.max(x1, dim=1)[0]  # (B, 128)
        
        x2 = self.layer2(rearrange(x1, 'b n d -> (b n) d'))
        x2 = rearrange(x2, '(b n) d -> b n d', b=B, n=N)
        f2 = torch.max(x2, dim=1)[0]  # (B, 256)
        
        x3 = self.layer3(rearrange(x2, 'b n d -> (b n) d'))
        x3 = rearrange(x3, '(b n) d -> b n d', b=B, n=N)
        f3 = torch.max(x3, dim=1)[0]  # (B, 512)
        
        # Global feature
        global_feat = self.global_pool(f3)  # (B, feature_dim)
        
        return global_feat


class RGBDPointCloudEncoder(nn.Module):
    """
    Encoder untuk RGB-D point cloud yang menggabungkan 3D coordinates dan RGB features
    """
    def __init__(self, feature_dim=256, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # Point coordinates encoder (x, y, z)
        self.coord_encoder = PointCloudEncoder(point_dim=3, feature_dim=128, num_points=num_points)
        
        # RGB features encoder
        self.rgb_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
    def forward(self, point_cloud, rgb_features=None):
        """
        Args:
            point_cloud: (B, N, 3) point coordinates
            rgb_features: (B, N, 3) RGB values (optional)
        Returns:
            features: (B, feature_dim) fused features
        """
        # Encode coordinates
        coord_feat = self.coord_encoder(point_cloud)  # (B, 128)
        
        if rgb_features is not None:
            # Encode RGB features
            B, N, _ = rgb_features.shape
            rgb_flat = rearrange(rgb_features, 'b n d -> (b n) d')
            rgb_feat_flat = self.rgb_encoder(rgb_flat)
            rgb_feat = rearrange(rgb_feat_flat, '(b n) d -> b n d', b=B, n=N)
            rgb_feat = torch.max(rgb_feat, dim=1)[0]  # (B, 128)
        else:
            rgb_feat = torch.zeros_like(coord_feat)
        
        # Fuse features
        fused = torch.cat([coord_feat, rgb_feat], dim=-1)  # (B, 256)
        output = self.fusion(fused)  # (B, feature_dim)
        
        return output
