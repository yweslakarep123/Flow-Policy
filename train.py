"""
Training script untuk FlowPolicy dengan RoboSet dataset
Kompatibel dengan struktur original FlowPolicy
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy.flowpolicy import FlowPolicy
from model.common.normalizer import LinearNormalizer
from utils.dataset_loader import load_robo_set_dataset
from common.replay_buffer import ReplayBuffer
from common.logger_util import setup_logger
from common.checkpoint_util import save_checkpoint, load_checkpoint


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    
    # Load dataset
    # Jika data_path tidak diberikan, gunakan HF cache
    if args.data_path is None:
        print('Data path tidak diberikan, menggunakan HuggingFace cache...')
        args.data_path = ""  # Empty string akan trigger HF cache lookup
    
    print(f'Loading dataset from {args.data_path if args.data_path else "HuggingFace cache"}...')
    dataset = load_robo_set_dataset(
        data_path=args.data_path if args.data_path else "",
        task_name=args.task_name,
        horizon=args.horizon,
        use_hf_cache=args.use_hf_cache,
        key_blacklist=args.key_blacklist
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create shape_meta for FlowPolicy
    # Get sample to infer shapes
    try:
        sample = dataset[0]
    except Exception as e:
        print(f"Error getting sample from dataset: {e}")
        print(f"Dataset length: {len(dataset)}")
        raise
    
    # Debug: print sample keys
    print(f"\nSample keys: {list(sample.keys())}")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"  {key}: dict with keys {list(value.keys())}")
            for sub_key, sub_value in value.items():
                if torch.is_tensor(sub_value):
                    print(f"    {sub_key}: shape={sub_value.shape}")
        elif torch.is_tensor(value):
            print(f"  {key}: shape={value.shape}")
    
    # Helper function to convert shape to tuple
    def get_shape_tensor(tensor_or_array):
        """Convert tensor/array shape to tuple"""
        if torch.is_tensor(tensor_or_array):
            return tuple(tensor_or_array.shape[1:])  # Remove sequence dimension
        elif isinstance(tensor_or_array, np.ndarray):
            return tuple(tensor_or_array.shape[1:])
        else:
            return None
    
    # Get action shape
    action_data = sample.get('action', sample.get('actions', None))
    if action_data is None:
        raise ValueError("No 'action' or 'actions' key found in sample!")
    action_shape = get_shape_tensor(action_data)
    
    shape_meta = {
        'action': {
            'shape': action_shape
        },
        'obs': {}
    }
    
    # Parse observation shapes
    if 'obs' in sample:
        obs_data = sample['obs']
        if isinstance(obs_data, dict):
            for key, value in obs_data.items():
                shape = get_shape_tensor(value)
                if shape is not None:
                    shape_meta['obs'][key] = {'shape': shape}
        else:
            shape = get_shape_tensor(obs_data)
            if shape is not None:
                shape_meta['obs']['obs'] = {'shape': shape}
    
    # Ensure point_cloud and agent_pos are in shape_meta
    if 'point_cloud' not in shape_meta['obs']:
        # Try to infer from sample
        if 'obs' in sample and isinstance(sample['obs'], dict):
            if 'point_cloud' in sample['obs']:
                shape = get_shape_tensor(sample['obs']['point_cloud'])
                if shape is not None:
                    shape_meta['obs']['point_cloud'] = {'shape': shape}
                else:
                    shape_meta['obs']['point_cloud'] = {'shape': (args.num_points, 3)}
            else:
                # Default point cloud shape
                shape_meta['obs']['point_cloud'] = {'shape': (args.num_points, 3)}
        else:
            # Default point cloud shape
            shape_meta['obs']['point_cloud'] = {'shape': (args.num_points, 3)}
    
    if 'agent_pos' not in shape_meta['obs']:
        # Try to infer or use default
        if 'obs' in sample and isinstance(sample['obs'], dict) and 'agent_pos' in sample['obs']:
            shape = get_shape_tensor(sample['obs']['agent_pos'])
            if shape is not None:
                shape_meta['obs']['agent_pos'] = {'shape': shape}
            else:
                shape_meta['obs']['agent_pos'] = {'shape': (9,)}
        else:
            # Try to infer from action shape (usually same as robot state)
            if action_shape and len(action_shape) == 1:
                shape_meta['obs']['agent_pos'] = {'shape': action_shape}
            else:
                # Default robot state dimension
                shape_meta['obs']['agent_pos'] = {'shape': (9,)}  # 7 arm + 2 gripper
    
    # Convert all shapes to tuples (not torch.Size)
    def convert_shape_meta(meta):
        """Recursively convert shape_meta to use tuples instead of torch.Size"""
        if isinstance(meta, dict):
            result = {}
            for key, value in meta.items():
                if key == 'shape' and isinstance(value, (torch.Size, tuple)):
                    # Convert torch.Size to tuple
                    result[key] = tuple(value)
                else:
                    result[key] = convert_shape_meta(value)
            return result
        return meta
    
    shape_meta = convert_shape_meta(shape_meta)
    
    # IMPORTANT: dict_apply with lambda x: x['shape'] expects structure like:
    # obs_shape_meta = {'obs': SomeObj, 'point_cloud': SomeObj, ...}
    # where SomeObj has a 'shape' attribute/key.
    # But dict_apply recurses into nested dicts, so if SomeObj is {'shape': (75,)},
    # it will recurse into it and try to apply lambda to (75,), causing error.
    # 
    # The correct structure is to have obs_shape_meta values DIRECTLY be the shapes:
    # obs_shape_meta = {'obs': (75,), 'point_cloud': (1024, 3), ...}
    # Then dict_apply(obs_shape_meta, lambda x: x) returns the same dict.
    
    # Flatten the obs structure: extract 'shape' from each nested dict
    if 'obs' in shape_meta:
        obs_meta = shape_meta['obs']
        if isinstance(obs_meta, dict):
            flattened_obs = {}
            for key, value in obs_meta.items():
                if isinstance(value, dict) and 'shape' in value:
                    # Extract shape from nested dict
                    flattened_obs[key] = tuple(value['shape']) if isinstance(value['shape'], (torch.Size, tuple, list)) else value['shape']
                elif isinstance(value, (tuple, torch.Size, list)):
                    # Already a shape
                    flattened_obs[key] = tuple(value)
                else:
                    # Unknown format, keep as is
                    flattened_obs[key] = value
            shape_meta['obs'] = flattened_obs
    
    print(f'Shape meta: {shape_meta}')
    
    # Debug: print what dict_apply will see
    print(f'\nDebug - obs_shape_meta structure (before dict_apply):')
    if 'obs' in shape_meta:
        def print_nested_dict(d, indent=0, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            for key, value in d.items():
                prefix = '  ' * indent
                if isinstance(value, dict):
                    print(f"{prefix}{key}: dict")
                    if 'shape' in value:
                        print(f"{prefix}  shape: {type(value['shape']).__name__} = {value['shape']}")
                    else:
                        print_nested_dict(value, indent + 1, max_depth, current_depth + 1)
                else:
                    print(f"{prefix}{key}: {type(value).__name__} = {value}")
        print_nested_dict(shape_meta['obs'])
    
    # Create FlowPolicy model
    print('Creating FlowPolicy model...')
    
    # Point cloud encoder config
    pointcloud_encoder_cfg = {
        'in_channels': 6 if args.use_pc_color else 3,
        'out_channels': args.encoder_output_dim,
        'use_layernorm': args.use_layernorm,
        'final_norm': args.final_norm if hasattr(args, 'final_norm') else 'none'
    }
    
    policy = FlowPolicy(
        shape_meta=shape_meta,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        obs_as_global_cond=args.obs_as_global_cond,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=args.down_dims,
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        condition_type=args.condition_type,
        use_down_condition=args.use_down_condition,
        use_mid_condition=args.use_mid_condition,
        use_up_condition=args.use_up_condition,
        encoder_output_dim=args.encoder_output_dim,
        crop_shape=args.crop_shape,
        use_pc_color=args.use_pc_color,
        pointnet_type=args.pointnet_type,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        Conditional_ConsistencyFM={
            'eps': args.eps,
            'num_segments': args.num_segments,
            'boundary': args.boundary,
            'delta': args.delta,
            'alpha': args.alpha,
            'num_inference_step': args.num_inference_step
        },
        eta=args.eta
    ).to(device)
    
    # Setup normalizer
    print('Setting up normalizer...')
    normalizer = LinearNormalizer()
    
    # Fit normalizer on dataset
    all_obs = []
    all_actions = []
    for i in range(min(1000, len(dataset))):  # Sample for normalizer
        sample = dataset[i]
        if 'obs' in sample:
            if isinstance(sample['obs'], dict):
                all_obs.append({k: v.numpy() for k, v in sample['obs'].items()})
            else:
                all_obs.append(sample['obs'].numpy())
        if 'action' in sample:
            all_actions.append(sample['action'].numpy())
    
    # Check for NaN/Inf in data before fitting
    def check_data_validity(data, name):
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                result[k] = check_data_validity(v, f"{name}.{k}")
            return result
        elif isinstance(data, np.ndarray):
            data = data.copy()  # Make a copy to avoid modifying original
            if np.any(np.isnan(data)):
                print(f"WARNING: NaN found in {name}, replacing with 0")
                data = np.nan_to_num(data, nan=0.0)
            if np.any(np.isinf(data)):
                print(f"WARNING: Inf found in {name}, clipping to [-1e6, 1e6]")
                data = np.clip(data, -1e6, 1e6)
            return data
        else:
            return data
    
    if all_actions:
        action_data = np.concatenate(all_actions, axis=0)
        action_data = check_data_validity(action_data, 'action')
        print(f"\nFitting normalizer for actions...")
        print(f"  Shape: {action_data.shape}")
        print(f"  Stats: min={action_data.min():.4f}, max={action_data.max():.4f}, mean={action_data.mean():.4f}, std={action_data.std():.4f}")
        normalizer.fit({'action': action_data})
    
    if all_obs:
        if isinstance(all_obs[0], dict):
            obs_dict = {}
            for key in all_obs[0].keys():
                obs_data = np.concatenate([o[key] for o in all_obs], axis=0)
                obs_data = check_data_validity(obs_data, f'obs.{key}')
                obs_dict[key] = obs_data
                print(f"Obs.{key} stats: min={obs_data.min():.4f}, max={obs_data.max():.4f}, mean={obs_data.mean():.4f}, std={obs_data.std():.4f}")
            normalizer.fit({'obs': obs_dict})
        else:
            obs_data = np.concatenate(all_obs, axis=0)
            obs_data = check_data_validity(obs_data, 'obs')
            normalizer.fit({'obs': obs_data})
            print(f"Obs stats: min={obs_data.min():.4f}, max={obs_data.max():.4f}, mean={obs_data.mean():.4f}, std={obs_data.std():.4f}")
    
    # Move normalizer parameters to device
    # ParameterDict stores Parameter objects which are Tensors
    # We need to move all parameters recursively
    def move_paramdict_to_device(pd, dev):
        """Recursively move ParameterDict parameters to device"""
        if isinstance(pd, torch.nn.ParameterDict):
            for key in list(pd.keys()):
                value = pd[key]
                if isinstance(value, torch.nn.ParameterDict):
                    move_paramdict_to_device(value, dev)
                elif isinstance(value, (torch.Tensor, torch.nn.Parameter)):
                    # Replace with Parameter on correct device
                    new_param = torch.nn.Parameter(value.data.to(dev), requires_grad=False)
                    pd[key] = new_param
    
    if hasattr(normalizer, 'params_dict'):
        for key, value in normalizer.params_dict.items():
            if isinstance(value, torch.nn.ParameterDict):
                move_paramdict_to_device(value, device)
            elif isinstance(value, torch.Tensor):
                normalizer.params_dict[key] = value.to(device)
    
    print('Normalizer parameters moved to device')
    
    # Validate normalizer parameters (check for NaN/Inf)
    def validate_normalizer_params(pd, path=""):
        if isinstance(pd, torch.nn.ParameterDict):
            for key, value in pd.items():
                if isinstance(value, torch.nn.ParameterDict):
                    validate_normalizer_params(value, f"{path}.{key}" if path else key)
                elif isinstance(value, torch.Tensor):
                    if torch.any(torch.isnan(value)):
                        print(f"ERROR: NaN in normalizer param at {path}.{key}")
                        raise ValueError(f"NaN in normalizer parameter {path}.{key}")
                    if torch.any(torch.isinf(value)):
                        print(f"ERROR: Inf in normalizer param at {path}.{key}")
                        raise ValueError(f"Inf in normalizer parameter {path}.{key}")
                    # Check for very small scale values that could cause issues
                    if 'scale' in key.lower() and torch.any(torch.abs(value) < 1e-8):
                        print(f"WARNING: Very small scale values at {path}.{key}, min={torch.abs(value).min()}")
    
    if hasattr(normalizer, 'params_dict'):
        for key, value in normalizer.params_dict.items():
            if isinstance(value, torch.nn.ParameterDict):
                validate_normalizer_params(value, key)
    
    policy.set_normalizer(normalizer)
    
    # Create optimizer
    optimizer = optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min
    )
    
    # Create tensorboard writer
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print('Starting training...')
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        policy.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            # Move batch to device (recursively handle nested dicts)
            def move_to_device(obj):
                if isinstance(obj, dict):
                    return {k: move_to_device(v) for k, v in obj.items()}
                elif torch.is_tensor(obj):
                    return obj.to(device)
                elif isinstance(obj, np.ndarray):
                    # Convert numpy array to tensor and move to device
                    return torch.from_numpy(obj).to(device)
                elif isinstance(obj, (list, tuple)):
                    return type(obj)([move_to_device(item) for item in obj])
                else:
                    return obj
            
            batch_device = move_to_device(batch)
            
            # Debug: verify all tensors are on correct device
            def check_device(obj, path=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        check_device(v, f"{path}.{k}" if path else k)
                elif torch.is_tensor(obj):
                    if obj.device != device:
                        print(f"ERROR: Tensor at {path} is on {obj.device}, expected {device}, shape={obj.shape}")
                elif isinstance(obj, (list, tuple)):
                    for i, item in enumerate(obj):
                        check_device(item, f"{path}[{i}]")
                elif isinstance(obj, np.ndarray):
                    print(f"WARNING: Numpy array at {path} not converted to tensor, shape={obj.shape}")
            
            # Check device for first batch only
            if epoch == 0 and num_batches == 0:
                print("\n=== Checking device placement ===")
                check_device(batch_device)
                print("=== End device check ===\n")
            
            optimizer.zero_grad()
            
            # Check for NaN/Inf in batch before computing loss
            def check_batch_validity(batch_data, path=""):
                if isinstance(batch_data, dict):
                    for k, v in batch_data.items():
                        check_batch_validity(v, f"{path}.{k}" if path else k)
                elif torch.is_tensor(batch_data):
                    if torch.any(torch.isnan(batch_data)):
                        print(f"ERROR: NaN found in batch at {path}")
                        return False
                    if torch.any(torch.isinf(batch_data)):
                        print(f"ERROR: Inf found in batch at {path}")
                        return False
                return True
            
            if not check_batch_validity(batch_device):
                print("Skipping batch due to NaN/Inf")
                continue
            
            # Compute loss
            loss, loss_dict = policy.compute_loss(batch_device)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nERROR: Loss is NaN/Inf: {loss.item()}")
                print(f"  Loss dict: {loss_dict}")
                print(f"  Epoch: {epoch+1}, Batch: {num_batches}")
                # Try to get more info
                with torch.no_grad():
                    # Check normalized data
                    try:
                        nobs = policy.normalizer.normalize(batch_device['obs'])
                        nactions = policy.normalizer['action'].normalize(batch_device['action'])
                        
                        obs_has_nan = False
                        if isinstance(nobs, dict):
                            for k, v in nobs.items():
                                if torch.is_tensor(v) and torch.any(torch.isnan(v)):
                                    print(f"  Normalized obs.{k} has NaN")
                                    obs_has_nan = True
                        else:
                            if torch.is_tensor(nobs) and torch.any(torch.isnan(nobs)):
                                print(f"  Normalized obs has NaN")
                                obs_has_nan = True
                        
                        if torch.any(torch.isnan(nactions)):
                            print(f"  Normalized actions has NaN")
                        
                        # Check raw data ranges
                        if 'obs' in batch_device:
                            if isinstance(batch_device['obs'], dict):
                                for k, v in batch_device['obs'].items():
                                    if torch.is_tensor(v):
                                        print(f"  Raw obs.{k}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                            else:
                                v = batch_device['obs']
                                if torch.is_tensor(v):
                                    print(f"  Raw obs: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                        
                        if 'action' in batch_device:
                            v = batch_device['action']
                            if torch.is_tensor(v):
                                print(f"  Raw action: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                    except Exception as e:
                        print(f"  Error checking normalized data: {e}")
                
                print("  Suggestion: Try reducing learning rate (--lr 1e-5) or check data normalization\n")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in policy.named_parameters():
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)):
                        print(f"WARNING: NaN gradient in {name}, zeroing it")
                        param.grad.zero_()
                        has_nan_grad = True
                    if torch.any(torch.isinf(param.grad)):
                        print(f"WARNING: Inf gradient in {name}, clipping it")
                        param.grad = torch.clamp(param.grad, -1e6, 1e6)
                        has_nan_grad = True
            
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Log to tensorboard
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f'Loss/{key}', value, global_step)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        writer.add_scalar('Loss/TrainAvg', avg_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        scheduler.step()
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(
                checkpoint_path,
                policy,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                args
            )
            print(f'Saved best model (loss={best_loss:.4f}) to {checkpoint_path}')
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_checkpoint(
                checkpoint_path,
                policy,
                optimizer,
                scheduler,
                epoch,
                avg_loss,
                args
            )
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FlowPolicy with RoboSet dataset')
    
    # Data args
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to RoboSet dataset folder. Jika tidak diberikan atau tidak ditemukan, akan otomatis mencari di HuggingFace cache (~/.cache/huggingface/hub/datasets--jdvakil--RoboSet_Sim)')
    parser.add_argument('--task_name', type=str, default=None,
                       help='Task name (optional), e.g., FK1_MicroOpenRandom_v2d-v4')
    parser.add_argument('--key_blacklist', nargs='+', default=[],
                       help='Keys to exclude from dataset')
    parser.add_argument('--use_hf_cache', type=bool, default=True,
                       help='Gunakan HuggingFace cache jika data_path tidak ditemukan (default: True)')
    parser.add_argument('--horizon', type=int, default=16,
                       help='Horizon for action prediction')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='Number of points in point cloud')
    
    # Model args
    parser.add_argument('--n_action_steps', type=int, default=8,
                       help='Number of action steps')
    parser.add_argument('--n_obs_steps', type=int, default=1,
                       help='Number of observation steps')
    parser.add_argument('--obs_as_global_cond', type=bool, default=True,
                       help='Use observations as global condition')
    parser.add_argument('--diffusion_step_embed_dim', type=int, default=256,
                       help='Diffusion step embedding dimension')
    parser.add_argument('--down_dims', nargs='+', type=int, default=[256, 512, 1024],
                       help='Down dimensions for U-Net')
    parser.add_argument('--kernel_size', type=int, default=5,
                       help='Kernel size')
    parser.add_argument('--n_groups', type=int, default=8,
                       help='Number of groups for group norm')
    parser.add_argument('--condition_type', type=str, default='film',
                       help='Condition type: film, cross_attention, etc.')
    parser.add_argument('--use_down_condition', type=bool, default=True,
                       help='Use condition in downsampling')
    parser.add_argument('--use_mid_condition', type=bool, default=True,
                       help='Use condition in middle')
    parser.add_argument('--use_up_condition', type=bool, default=True,
                       help='Use condition in upsampling')
    parser.add_argument('--encoder_output_dim', type=int, default=256,
                       help='Encoder output dimension')
    parser.add_argument('--crop_shape', type=int, nargs=2, default=None,
                       help='Crop shape for images')
    parser.add_argument('--use_pc_color', action='store_true',
                       help='Use point cloud color')
    parser.add_argument('--pointnet_type', type=str, default='mlp',
                       help='PointNet type: mlp')
    parser.add_argument('--use_layernorm', action='store_true',
                       help='Use layer norm in PointNet')
    parser.add_argument('--final_norm', type=str, default='none',
                       choices=['none', 'layernorm'],
                       help='Final normalization')
    
    # Consistency Flow Matching args
    parser.add_argument('--eps', type=float, default=0.01,
                       help='Epsilon for CFM')
    parser.add_argument('--num_segments', type=int, default=2,
                       help='Number of segments')
    parser.add_argument('--boundary', type=float, default=1.0,
                       help='Boundary threshold')
    parser.add_argument('--delta', type=float, default=0.01,
                       help='Delta for CFM')
    parser.add_argument('--alpha', type=float, default=1e-5,
                       help='Alpha for velocity loss')
    parser.add_argument('--num_inference_step', type=int, default=1,
                       help='Number of inference steps')
    parser.add_argument('--eta', type=float, default=0.01,
                       help='Eta parameter')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='Minimum learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Checkpoint save frequency')
    
    # Logging args
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Convert list args
    if isinstance(args.down_dims, list) and len(args.down_dims) == 1:
        args.down_dims = tuple(map(int, args.down_dims[0].split(',')))
    elif isinstance(args.down_dims, list):
        args.down_dims = tuple(args.down_dims)
    
    train(args)
