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
        
        # Load all data
        self.data = []
        for file_path in self.data_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    data_dict = {}
                    # Load all keys
                    for key in f.keys():
                        if key not in self.key_blacklist:
                            data_dict[key] = np.array(f[key])
                    self.data.append(data_dict)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if len(self.data) == 0:
            raise ValueError("No valid data files found!")
        
        # Compute indices for sequences
        self.indices = []
        for i, data_dict in enumerate(self.data):
            # Get sequence length from 'action' or 'obs' key
            if 'action' in data_dict:
                seq_len = len(data_dict['action'])
            elif 'obs' in data_dict:
                seq_len = len(data_dict['obs'])
            else:
                # Try to get from first available key
                key = list(data_dict.keys())[0]
                seq_len = len(data_dict[key])
            
            # Create indices for this file
            for start_idx in range(seq_len - horizon + 1):
                self.indices.append((i, start_idx))
        
        print(f"Total sequences: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx, start_idx = self.indices[idx]
        data_dict = self.data[file_idx]
        
        # Extract sequence
        end_idx = start_idx + self.horizon
        
        sample = {}
        for key, value in data_dict.items():
            if key not in self.key_blacklist:
                # Handle different key formats
                if key == 'obs':
                    # Handle nested obs dict
                    if isinstance(value, dict):
                        obs_sample = {}
                        for obs_key, obs_value in value.items():
                            obs_sample[obs_key] = torch.FloatTensor(obs_value[start_idx:end_idx])
                        sample['obs'] = obs_sample
                    else:
                        sample['obs'] = torch.FloatTensor(value[start_idx:end_idx])
                elif key == 'action':
                    sample['action'] = torch.FloatTensor(value[start_idx:end_idx])
                else:
                    # Generic handling
                    try:
                        sample[key] = torch.FloatTensor(value[start_idx:end_idx])
                    except:
                        sample[key] = value[start_idx:end_idx]
        
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
