"""
Dataset loader untuk RoboSet_Sim .h5 files
Kompatibel dengan format RoboHive dataset
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import os
import glob


class RoboSetDataset(Dataset):
    """
    Dataset untuk loading RoboSet_Sim .h5 files
    """
    def __init__(
        self,
        data_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        key_blacklist: Optional[List[str]] = None,
    ):
        """
        Args:
            data_path: Path ke folder yang berisi .h5 files atau path ke file .h5
            horizon: Horizon untuk action prediction
            pad_before: Padding sebelum sequence
            pad_after: Padding setelah sequence
            key_blacklist: Keys yang tidak digunakan
        """
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.key_blacklist = key_blacklist or []
        
        # Load data files
        if os.path.isfile(data_path):
            self.data_files = [data_path]
        elif os.path.isdir(data_path):
            # Find all .h5 files recursively
            self.data_files = glob.glob(os.path.join(data_path, '**/*.h5'), recursive=True)
        else:
            raise ValueError(f"Data path tidak valid: {data_path}")
        
        print(f"Found {len(self.data_files)} data files")
        
        # Helper function to recursively load h5py groups
        def load_h5_group(group, prefix='', max_depth=10, current_depth=0):
            """Recursively load h5py groups and datasets"""
            if current_depth >= max_depth:
                return {}
            
            data = {}
            for key in group.keys():
                full_key = f"{prefix}/{key}" if prefix else key
                if key in self.key_blacklist:
                    continue
                
                item = group[key]
                if isinstance(item, h5py.Group):
                    # Check if this looks like a trial group (e.g., Trial0, Trial1, etc.)
                    if key.startswith('Trial') or key.startswith('trial'):
                        # Load trial data directly
                        trial_data = {}
                        for trial_key in item.keys():
                            trial_item = item[trial_key]
                            if isinstance(trial_item, h5py.Dataset):
                                trial_data[trial_key] = np.array(trial_item)
                            elif isinstance(trial_item, h5py.Group):
                                # Handle nested groups in trials (e.g., obs_dict)
                                for nested_key in trial_item.keys():
                                    nested_item = trial_item[nested_key]
                                    if isinstance(nested_item, h5py.Dataset):
                                        trial_data[nested_key] = np.array(nested_item)
                        
                        # Merge trial data into main data, handling multiple trials
                        for trial_data_key, trial_data_value in trial_data.items():
                            # Normalize key names
                            if trial_data_key == 'actions' or trial_data_key == 'action':
                                if 'action' not in data:
                                    data['action'] = []
                                data['action'].append(trial_data_value)
                            elif trial_data_key == 'observations' or trial_data_key == 'observation' or trial_data_key == 'obs':
                                if 'obs' not in data:
                                    data['obs'] = {}
                                # Handle obs as dict
                                if isinstance(trial_data_value, dict):
                                    for obs_key, obs_value in trial_data_value.items():
                                        if obs_key not in data['obs']:
                                            data['obs'][obs_key] = []
                                        data['obs'][obs_key].append(obs_value)
                                else:
                                    if 'obs' not in data['obs']:
                                        data['obs']['obs'] = []
                                    data['obs']['obs'].append(trial_data_value)
                            else:
                                # Store other keys
                                if trial_data_key not in data:
                                    data[trial_data_key] = []
                                data[trial_data_key].append(trial_data_value)
                    else:
                        # Recursively load subgroup
                        sub_data = load_h5_group(item, full_key, max_depth, current_depth + 1)
                        data.update(sub_data)
                elif isinstance(item, h5py.Dataset):
                    # Load dataset
                    try:
                        data[full_key] = np.array(item)
                    except Exception as e:
                        print(f"Warning: Could not load {full_key}: {e}")
            return data
        
        # Load all data
        self.data = []
        for file_path in self.data_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    # Try to load from root level first
                    data_dict = {}
                    
                    # Check if it's a RoboHive format with trials
                    trial_keys = [k for k in f.keys() if k.startswith('Trial') or k.startswith('trial')]
                    
                    if trial_keys:
                        # RoboHive format with multiple trials
                        print(f"  Found {len(trial_keys)} trials in {os.path.basename(file_path)}")
                        all_actions = []
                        all_obs = {}
                        
                        for trial_key in sorted(trial_keys):
                            trial_group = f[trial_key]
                            
                            # Load actions
                            if 'actions' in trial_group:
                                trial_actions = np.array(trial_group['actions'])
                                all_actions.append(trial_actions)
                            elif 'action' in trial_group:
                                trial_actions = np.array(trial_group['action'])
                                all_actions.append(trial_actions)
                            
                            # Load observations
                            if 'observations' in trial_group:
                                obs_group = trial_group['observations']
                                if isinstance(obs_group, h5py.Group):
                                    # Nested obs structure
                                    for obs_key in obs_group.keys():
                                        if obs_key not in all_obs:
                                            all_obs[obs_key] = []
                                        all_obs[obs_key].append(np.array(obs_group[obs_key]))
                                else:
                                    # Direct obs array
                                    if 'obs' not in all_obs:
                                        all_obs['obs'] = []
                                    all_obs['obs'].append(np.array(obs_group))
                            elif 'observation' in trial_group:
                                obs_data = np.array(trial_group['observation'])
                                if 'obs' not in all_obs:
                                    all_obs['obs'] = []
                                all_obs['obs'].append(obs_data)
                            elif 'obs' in trial_group:
                                obs_group = trial_group['obs']
                                if isinstance(obs_group, h5py.Group):
                                    for obs_key in obs_group.keys():
                                        if obs_key not in all_obs:
                                            all_obs[obs_key] = []
                                        all_obs[obs_key].append(np.array(obs_group[obs_key]))
                                else:
                                    if 'obs' not in all_obs:
                                        all_obs['obs'] = []
                                    all_obs['obs'].append(np.array(obs_group))
                        
                        # Concatenate all trials
                        if all_actions:
                            data_dict['action'] = np.concatenate(all_actions, axis=0)
                        
                        # Concatenate obs
                        if all_obs:
                            data_dict['obs'] = {}
                            for obs_key, obs_list in all_obs.items():
                                try:
                                    data_dict['obs'][obs_key] = np.concatenate(obs_list, axis=0)
                                except Exception as e:
                                    print(f"  Warning: Could not concatenate {obs_key}: {e}")
                                    # Try to take first one
                                    data_dict['obs'][obs_key] = obs_list[0]
                    
                    # Check if it's a RoboHive format (has 'data' group)
                    elif 'data' in f.keys() and isinstance(f['data'], h5py.Group):
                        # RoboHive format: data is in 'data' group
                        data_dict = load_h5_group(f['data'])
                        # Also check for 'timeouts' or other metadata
                        if 'timeouts' in f.keys():
                            data_dict['timeouts'] = np.array(f['timeouts'])
                    else:
                        # Direct format: load from root
                        data_dict = load_h5_group(f)
                    
                    # Debug: print structure
                    if len(self.data) == 0:  # Only print for first file
                        print(f"\nDataset structure (first file: {os.path.basename(file_path)}):")
                        print(f"  Keys: {list(data_dict.keys())}")
                        for key, value in list(data_dict.items())[:5]:  # Print first 5 keys
                            if isinstance(value, np.ndarray):
                                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                            else:
                                print(f"  {key}: type={type(value)}")
                    
                    # Concatenate multiple trials if they were stored as lists
                    for key in list(data_dict.keys()):
                        if isinstance(data_dict[key], list):
                            # Concatenate along first dimension
                            try:
                                data_dict[key] = np.concatenate(data_dict[key], axis=0)
                            except:
                                # If concatenation fails, keep as list
                                pass
                    
                    # Handle nested obs structure
                    if 'obs' in data_dict and isinstance(data_dict['obs'], dict):
                        # Concatenate obs dict values if they are lists
                        for obs_key in list(data_dict['obs'].keys()):
                            if isinstance(data_dict['obs'][obs_key], list):
                                try:
                                    data_dict['obs'][obs_key] = np.concatenate(data_dict['obs'][obs_key], axis=0)
                                except:
                                    pass
                    
                    if 'obs' not in data_dict:
                        # Try to find observation keys
                        obs_keys = [k for k in data_dict.keys() if 'obs' in k.lower() or 'point' in k.lower() or 'image' in k.lower()]
                        if obs_keys:
                            print(f"  Found observation-like keys: {obs_keys}")
                    
                    self.data.append(data_dict)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(self.data) == 0:
            raise ValueError("No valid data files found!")
        
        # Compute indices for sequences
        self.indices = []
        for i, data_dict in enumerate(self.data):
            # Get sequence length from 'action' or 'obs' key
            seq_len = None
            
            # Try different possible keys for sequence length
            if 'action' in data_dict:
                action_data = data_dict['action']
                if isinstance(action_data, list):
                    # Sum lengths of all trials
                    seq_len = sum(len(trial) for trial in action_data)
                else:
                    seq_len = len(action_data)
            elif 'actions' in data_dict:
                action_data = data_dict['actions']
                if isinstance(action_data, list):
                    seq_len = sum(len(trial) for trial in action_data)
                else:
                    seq_len = len(action_data)
            elif 'obs' in data_dict:
                obs_data = data_dict['obs']
                if isinstance(obs_data, dict):
                    # Get length from first obs key
                    first_key = list(obs_data.keys())[0]
                    seq_len = len(obs_data[first_key])
                else:
                    seq_len = len(obs_data)
            else:
                # Try to find any key with sequence dimension
                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray) and len(value.shape) > 0:
                        seq_len = len(value)
                        print(f"  Using {key} for sequence length: {seq_len}")
                        break
            
            if seq_len is None:
                print(f"Warning: Could not determine sequence length for file {i}")
                print(f"  Available keys: {list(data_dict.keys())}")
                continue
            
            if seq_len < horizon:
                print(f"Warning: Sequence length ({seq_len}) < horizon ({horizon}) for file {i}, skipping")
                continue
            
            # Create indices for this file
            num_sequences = seq_len - horizon + 1
            for start_idx in range(num_sequences):
                self.indices.append((i, start_idx))
            
            if i == 0:  # Print info for first file
                print(f"  File {i}: seq_len={seq_len}, num_sequences={num_sequences}")
        
        print(f"Total sequences: {len(self.indices)}")
        
        if len(self.indices) == 0:
            raise ValueError(
                f"No valid sequences found! "
                f"Check that data has sequence length >= horizon ({horizon}). "
                f"Available keys in first file: {list(self.data[0].keys()) if self.data else 'N/A'}"
            )
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx, start_idx = self.indices[idx]
        data_dict = self.data[file_idx]
        
        # Extract sequence
        end_idx = start_idx + self.horizon
        
        sample = {}
        obs_dict = {}
        
        for key, value in data_dict.items():
            if key in self.key_blacklist:
                continue
            
            # Handle nested keys (e.g., 'data/obs/point_cloud')
            if '/' in key:
                # Split into main key and subkey
                parts = key.split('/')
                main_key = parts[0]
                sub_key = '/'.join(parts[1:])
                
                if main_key == 'obs' or 'obs' in main_key.lower():
                    if main_key not in obs_dict:
                        obs_dict[main_key] = {}
                    try:
                        obs_dict[main_key][sub_key] = torch.FloatTensor(value[start_idx:end_idx])
                    except:
                        obs_dict[main_key][sub_key] = value[start_idx:end_idx]
                else:
                    # Store with original key
                    try:
                        sample[key] = torch.FloatTensor(value[start_idx:end_idx])
                    except:
                        sample[key] = value[start_idx:end_idx]
            else:
                # Handle direct keys
                if key == 'obs' or 'obs' in key.lower():
                    if isinstance(value, dict):
                        obs_sample = {}
                        for obs_key, obs_value in value.items():
                            try:
                                obs_sample[obs_key] = torch.FloatTensor(obs_value[start_idx:end_idx])
                            except:
                                obs_sample[obs_key] = obs_value[start_idx:end_idx]
                        obs_dict[key] = obs_sample
                    else:
                        try:
                            obs_dict[key] = torch.FloatTensor(value[start_idx:end_idx])
                        except:
                            obs_dict[key] = value[start_idx:end_idx]
                elif key == 'action' or key == 'actions':
                    # Normalize action key
                    sample['action'] = torch.FloatTensor(value[start_idx:end_idx])
                else:
                    # Generic handling
                    try:
                        sample[key] = torch.FloatTensor(value[start_idx:end_idx])
                    except:
                        sample[key] = value[start_idx:end_idx]
        
        # Combine obs dicts
        if obs_dict:
            if len(obs_dict) == 1 and 'obs' in obs_dict:
                sample['obs'] = obs_dict['obs']
            else:
                # Merge all obs dicts
                merged_obs = {}
                for k, v in obs_dict.items():
                    if isinstance(v, dict):
                        merged_obs.update(v)
                    else:
                        merged_obs[k] = v
                sample['obs'] = merged_obs
        
        # Ensure point_cloud and agent_pos are in obs dict
        if 'obs' in sample:
            obs = sample['obs']
            if not isinstance(obs, dict):
                # Convert to dict if it's not already
                obs = {'obs': obs}
                sample['obs'] = obs
            
            # Add point_cloud if not present
            if 'point_cloud' not in obs:
                # Generate synthetic point cloud (placeholder)
                # In real implementation, this should come from actual point cloud data
                # Get batch size from first obs key
                first_key = list(obs.keys())[0]
                first_value = obs[first_key]
                if isinstance(first_value, torch.Tensor):
                    batch_size = first_value.shape[0]
                elif isinstance(first_value, np.ndarray):
                    batch_size = first_value.shape[0]
                else:
                    batch_size = 1
                
                num_points = 1024
                # Generate random point cloud as placeholder
                # Shape: (batch_size, num_points, 3)
                point_cloud = torch.randn(batch_size, num_points, 3) * 0.5
                point_cloud[:, :, 2] = torch.abs(point_cloud[:, :, 2]) + 0.5  # Ensure z > 0
                obs['point_cloud'] = point_cloud
            
            # Add agent_pos if not present
            if 'agent_pos' not in obs:
                # Get batch size from first obs key
                first_key = list(obs.keys())[0]
                first_value = obs[first_key]
                if isinstance(first_value, torch.Tensor):
                    batch_size = first_value.shape[0]
                elif isinstance(first_value, np.ndarray):
                    batch_size = first_value.shape[0]
                else:
                    batch_size = 1
                
                # Try to extract from action or use default
                if 'action' in sample:
                    # Use action as agent_pos (robot state)
                    action = sample['action']
                    if isinstance(action, torch.Tensor):
                        if action.shape[-1] >= 9:
                            obs['agent_pos'] = action[..., :9]
                        else:
                            # Pad with zeros if needed
                            agent_pos = torch.zeros(batch_size, 9)
                            agent_pos[..., :action.shape[-1]] = action
                            obs['agent_pos'] = agent_pos
                    else:
                        obs['agent_pos'] = torch.zeros(batch_size, 9)
                elif 'obs' in obs:
                    # Try to extract first 9 dimensions from obs as agent_pos
                    obs_data = obs['obs']
                    if isinstance(obs_data, torch.Tensor):
                        if obs_data.shape[-1] >= 9:
                            obs['agent_pos'] = obs_data[..., :9]
                        else:
                            # Pad with zeros if needed
                            agent_pos = torch.zeros(batch_size, 9)
                            agent_pos[..., :obs_data.shape[-1]] = obs_data
                            obs['agent_pos'] = agent_pos
                    else:
                        obs['agent_pos'] = torch.zeros(batch_size, 9)
                else:
                    # Default: create zero tensor
                    obs['agent_pos'] = torch.zeros(batch_size, 9)
        
        return sample


def load_robo_set_dataset(
    data_path: str,
    task_name: Optional[str] = None,
    horizon: int = 16,
    **kwargs
):
    """
    Load RoboSet dataset
    
    Args:
        data_path: Path ke dataset folder
        task_name: Nama task (optional), contoh: 'FK1_MicroOpenRandom_v2d-v4'
        horizon: Horizon untuk action prediction
    
    Returns:
        dataset: RoboSetDataset instance
    """
    if task_name:
        # Construct path to task-specific folder
        task_path = os.path.join(data_path, task_name)
        if not os.path.exists(task_path):
            # Try to find in subdirectories
            task_path = None
            for root, dirs, files in os.walk(data_path):
                if task_name in root:
                    task_path = root
                    break
            if task_path is None:
                raise ValueError(f"Task {task_name} tidak ditemukan di {data_path}")
        data_path = task_path
    
    dataset = RoboSetDataset(data_path, horizon=horizon, **kwargs)
    return dataset
