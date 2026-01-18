"""
Utility functions untuk data processing
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple


class ExpertDataset(Dataset):
    """
    Dataset untuk expert demonstrations
    """
    def __init__(
        self,
        observations: List[np.ndarray],  # List of point clouds (N, 3)
        actions: List[np.ndarray],  # List of actions (action_dim,)
        rgb_features: List[np.ndarray] = None,  # Optional RGB features
        num_points: int = 1024
    ):
        self.observations = observations
        self.actions = actions
        self.rgb_features = rgb_features
        self.num_points = num_points
        
        # Normalize point clouds
        self._normalize_pointclouds()
        
    def _normalize_pointclouds(self):
        """Normalize point clouds ke [-1, 1] range"""
        all_points = np.concatenate(self.observations, axis=0)
        self.pc_mean = np.mean(all_points, axis=0, keepdims=True)
        self.pc_std = np.std(all_points, axis=0, keepdims=True) + 1e-8
        
        # Normalize
        for i in range(len(self.observations)):
            self.observations[i] = (self.observations[i] - self.pc_mean) / self.pc_std
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = torch.FloatTensor(self.observations[idx])
        action = torch.FloatTensor(self.actions[idx])
        
        sample = {
            'point_cloud': obs,
            'action': action
        }
        
        if self.rgb_features is not None:
            sample['rgb_features'] = torch.FloatTensor(self.rgb_features[idx])
        
        return sample


def collect_expert_demonstrations(
    env,
    policy_fn,
    num_episodes=50,
    max_steps=500
):
    """
    Collect expert demonstrations dari environment
    
    Args:
        env: environment
        policy_fn: expert policy function (obs -> action)
        num_episodes: number of episodes to collect
        max_steps: maximum steps per episode
    Returns:
        observations: list of point clouds
        actions: list of actions
        rgb_features: list of RGB features (if available)
    """
    observations = []
    actions = []
    rgb_features = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        point_cloud = obs['point_cloud']
        rgb_feat = obs.get('rgb_features', None)
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Get action from expert policy
            action = policy_fn(obs)
            
            # Store data
            observations.append(point_cloud.copy())
            actions.append(action.copy())
            if rgb_feat is not None:
                rgb_features.append(rgb_feat.copy())
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            point_cloud = obs['point_cloud']
            rgb_feat = obs.get('rgb_features', None)
            step += 1
    
    if len(rgb_features) == 0:
        rgb_features = None
    
    return observations, actions, rgb_features
