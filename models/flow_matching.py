"""
Consistency Flow Matching Model
Implementasi berdasarkan paper FlowPolicy untuk single-step inference
"""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding untuk conditioning"""
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: (B,) time values [0, 1]
        Returns:
            embed: (B, dim) time embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class VelocityNetwork(nn.Module):
    """
    Neural network untuk memprediksi velocity field
    Input: action + time + condition, Output: velocity
    """
    def __init__(self, action_dim, condition_dim, hidden_dim=512, time_dim=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Input: action + time_embedding + condition
        input_dim = action_dim + time_dim + condition_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, action, t, condition):
        """
        Args:
            action: (B, action_dim) current action
            t: (B,) time values
            condition: (B, condition_dim) conditional features (e.g., point cloud features)
        Returns:
            velocity: (B, action_dim) predicted velocity
        """
        t_emb = self.time_embedding(t)  # (B, time_dim)
        x = torch.cat([action, t_emb, condition], dim=-1)  # (B, action_dim + time_dim + condition_dim)
        velocity = self.network(x)  # (B, action_dim)
        return velocity


class ConsistencyFlowMatching(nn.Module):
    """
    Consistency Flow Matching model untuk FlowPolicy
    Menggunakan straight-line flows dengan velocity consistency
    """
    def __init__(self, action_dim, condition_dim, hidden_dim=512):
        super().__init__()
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        
        # Velocity network
        self.velocity_network = VelocityNetwork(
            action_dim=action_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim
        )
        
    def sample_noise(self, batch_size, device):
        """Sample noise dari distribusi awal (Gaussian)"""
        return torch.randn(batch_size, self.action_dim, device=device)
    
    def compute_flow(self, action, t, condition):
        """
        Compute flow/velocity untuk titik tertentu
        
        Args:
            action: (B, action_dim) titik dalam action space
            t: (B,) time values [0, 1]
            condition: (B, condition_dim) conditional features
        Returns:
            velocity: (B, action_dim) velocity field
        """
        return self.velocity_network(action, t, condition)
    
    def compute_consistency_loss(self, action_0, action_1, condition, t_0, t_1):
        """
        Compute consistency loss untuk velocity field normalization
        
        Args:
            action_0: (B, action_dim) action pada time t_0
            action_1: (B, action_dim) action pada time t_1
            condition: (B, condition_dim) conditional features
            t_0, t_1: (B,) time values
        Returns:
            loss: consistency loss
        """
        # Prediksi velocity pada kedua titik
        v_0 = self.compute_flow(action_0, t_0, condition)
        v_1 = self.compute_flow(action_1, t_1, condition)
        
        # Untuk straight-line flow, velocity harus konsisten
        # Normalisasi self-consistency: v(t_0, x_0) ≈ v(t_1, x_1) saat di endpoint yang sama
        dt = t_1 - t_0
        
        # Interpolasi linear untuk straight-line flow
        alpha = t_1  # weight untuk endpoint
        expected_action_1 = action_0 + dt * v_0
        
        # Consistency loss: prediksi velocity harus konsisten dengan straight-line flow
        consistency_loss = F.mse_loss(action_1, expected_action_1)
        
        # Velocity normalization: velocity values harus normalized
        velocity_norm_loss = F.mse_loss(v_0, v_1)
        
        return consistency_loss + 0.1 * velocity_norm_loss
    
    def forward(self, action, t, condition):
        """
        Forward pass untuk training
        
        Args:
            action: (B, action_dim) ground truth action atau noisy action
            t: (B,) time values
            condition: (B, condition_dim) conditional features
        Returns:
            velocity: (B, action_dim) predicted velocity
        """
        return self.compute_flow(action, t, condition)
    
    @torch.no_grad()
    def generate(self, condition, num_steps=1):
        """
        Generate action dari noise dalam single step atau multi-step
        
        Args:
            condition: (B, condition_dim) conditional features
            num_steps: number of inference steps (1 untuk single-step)
        Returns:
            action: (B, action_dim) generated action
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # Sample noise
        action = self.sample_noise(batch_size, device)
        
        if num_steps == 1:
            # Single-step inference untuk consistency flow matching
            t = torch.ones(batch_size, device=device)
            velocity = self.compute_flow(action, t, condition)
            # Straight-line flow: x_1 = x_0 + v(x_1, t=1)
            # Iterasi untuk konsistensi
            for _ in range(3):  # Fixed-point iteration
                velocity = self.compute_flow(action, t, condition)
                action = velocity  # Update action
            return action
        else:
            # Multi-step inference (optional)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.ones(batch_size, device=device) * (i + 1) * dt
                velocity = self.compute_flow(action, t, condition)
                action = action + dt * velocity
            return action
