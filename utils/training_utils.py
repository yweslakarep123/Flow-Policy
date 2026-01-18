"""
Training utilities untuk FlowPolicy
"""
import torch
import numpy as np
from typing import Dict, List
import torch.nn.functional as F


def sample_timesteps(batch_size, device, min_t=0.0, max_t=1.0):
    """Sample random timesteps untuk training"""
    return torch.rand(batch_size, device=device) * (max_t - min_t) + min_t


def compute_flow_matching_loss(policy, batch, device):
    """
    Compute flow matching loss untuk batch
    
    Args:
        policy: FlowPolicy model
        batch: batch dictionary dengan keys ['point_cloud', 'action', 'rgb_features']
        device: device
    Returns:
        loss: flow matching loss
    """
    point_cloud = batch['point_cloud'].to(device)  # (B, N, 3)
    action_gt = batch['action'].to(device)  # (B, action_dim)
    rgb_features = batch.get('rgb_features', None)
    if rgb_features is not None:
        rgb_features = rgb_features.to(device)
    
    # Sample timesteps
    t = sample_timesteps(point_cloud.shape[0], device)
    
    # Compute loss
    loss = policy.compute_loss(point_cloud, action_gt, t, rgb_features)
    
    return loss


def evaluate_policy(policy, env, num_episodes=10, device='cuda'):
    """
    Evaluate policy performance
    
    Args:
        policy: FlowPolicy model
        env: environment
        num_episodes: number of evaluation episodes
        device: device
    Returns:
        metrics: dictionary dengan evaluation metrics
    """
    policy.eval()
    
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        max_steps = 500
        
        while not done and step < max_steps:
            # Get point cloud
            point_cloud = torch.FloatTensor(obs['point_cloud']).unsqueeze(0).to(device)  # (1, N, 3)
            rgb_features = None
            if 'rgb_features' in obs:
                rgb_features = torch.FloatTensor(obs['rgb_features']).unsqueeze(0).to(device)
            
            # Predict action
            with torch.no_grad():
                action = policy.predict(point_cloud, rgb_features, num_steps=1)
                action = action.cpu().numpy()[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
        
        total_rewards.append(total_reward)
        
        # Check success (adjust based on environment)
        if info.get('success', False) or total_reward > 0.9:
            success_count += 1
    
    metrics = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'success_rate': success_count / num_episodes
    }
    
    return metrics
