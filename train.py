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
    print(f'Loading dataset from {args.data_path}...')
    dataset = load_robo_set_dataset(
        data_path=args.data_path,
        task_name=args.task_name,
        horizon=args.horizon,
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
    sample = dataset[0]
    
    shape_meta = {
        'action': {
            'shape': sample['action'].shape[1:]  # Remove sequence dimension
        },
        'obs': {}
    }
    
    # Parse observation shapes
    if 'obs' in sample:
        if isinstance(sample['obs'], dict):
            for key, value in sample['obs'].items():
                shape_meta['obs'][key] = {'shape': value.shape[1:]}
        else:
            shape_meta['obs']['obs'] = {'shape': sample['obs'].shape[1:]}
    
    # Ensure point_cloud and agent_pos are in shape_meta
    if 'point_cloud' not in shape_meta['obs']:
        # Try to infer from sample
        if 'obs' in sample and isinstance(sample['obs'], dict):
            if 'point_cloud' in sample['obs']:
                shape_meta['obs']['point_cloud'] = {'shape': sample['obs']['point_cloud'].shape[1:]}
            else:
                # Default point cloud shape
                shape_meta['obs']['point_cloud'] = {'shape': (args.num_points, 3)}
        
    if 'agent_pos' not in shape_meta['obs']:
        # Try to infer or use default
        if 'obs' in sample and isinstance(sample['obs'], dict) and 'agent_pos' in sample['obs']:
            shape_meta['obs']['agent_pos'] = {'shape': sample['obs']['agent_pos'].shape[1:]}
        else:
            # Default robot state dimension (adjust based on your environment)
            shape_meta['obs']['agent_pos'] = {'shape': (9,)}  # 7 arm + 2 gripper
    
    print(f'Shape meta: {shape_meta}')
    
    # Create FlowPolicy model
    print('Creating FlowPolicy model...')
    
    # Point cloud encoder config
    from types import SimpleNamespace
    pointcloud_encoder_cfg = SimpleNamespace(
        in_channels=6 if args.use_pc_color else 3,
        out_channels=args.encoder_output_dim,
        use_layernorm=args.use_layernorm,
        final_norm=args.final_norm if hasattr(args, 'final_norm') else 'none'
    )
    
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
    
    if all_actions:
        normalizer.fit({'action': np.concatenate(all_actions, axis=0)})
    if all_obs:
        if isinstance(all_obs[0], dict):
            obs_dict = {}
            for key in all_obs[0].keys():
                obs_dict[key] = np.concatenate([o[key] for o in all_obs], axis=0)
            normalizer.fit({'obs': obs_dict})
        else:
            normalizer.fit({'obs': np.concatenate(all_obs, axis=0)})
    
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
            # Move batch to device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, dict):
                    batch_device[key] = {k: v.to(device) for k, v in value.items()}
                elif torch.is_tensor(value):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value
            
            optimizer.zero_grad()
            
            # Compute loss
            loss, loss_dict = policy.compute_loss(batch_device)
            
            # Backward pass
            loss.backward()
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
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to RoboSet dataset folder')
    parser.add_argument('--task_name', type=str, default=None,
                       help='Task name (optional), e.g., FK1_MicroOpenRandom_v2d-v4')
    parser.add_argument('--key_blacklist', nargs='+', default=[],
                       help='Keys to exclude from dataset')
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
