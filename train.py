"""
Training script untuk FlowPolicy
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os

from policies.flow_policy import FlowPolicy
from envs.franka_kitchen_wrapper import make_franka_kitchen_env
from utils.data_utils import ExpertDataset, collect_expert_demonstrations
from utils.training_utils import compute_flow_matching_loss, evaluate_policy


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create environment
    print('Creating environment...')
    env = make_franka_kitchen_env(
        env_name=args.env_name,
        tasks_to_complete=args.tasks,
        num_points=args.num_points,
        use_rgb=args.use_rgb
    )
    
    # Get action dimension
    action_dim = env.action_space.shape[0]
    print(f'Action dimension: {action_dim}')
    
    # Collect or load expert demonstrations
    if args.demo_path:
        # Load from file (implement sesuai kebutuhan)
        print(f'Loading demonstrations from {args.demo_path}')
        # Placeholder - implement data loading
        observations, actions, rgb_features = [], [], []
    else:
        print(f'Collecting {args.num_demos} expert demonstrations...')
        # Placeholder expert policy - gunakan random atau expert yang sudah ada
        def expert_policy(obs):
            # Random policy sebagai placeholder
            # Dalam implementasi nyata, gunakan expert policy yang sudah trained
            return env.action_space.sample()
        
        observations, actions, rgb_features = collect_expert_demonstrations(
            env, expert_policy, num_episodes=args.num_demos, max_steps=args.max_steps
        )
    
    # Create dataset
    print('Creating dataset...')
    dataset = ExpertDataset(observations, actions, rgb_features, num_points=args.num_points)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Create model
    print('Creating FlowPolicy model...')
    policy = FlowPolicy(
        action_dim=action_dim,
        point_cloud_dim=3,
        feature_dim=args.feature_dim,
        num_points=args.num_points,
        hidden_dim=args.hidden_dim,
        use_rgb=args.use_rgb
    ).to(device)
    
    # Create optimizer dengan weight decay untuk regularization
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    # Learning rate scheduler dengan warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create tensorboard writer
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    print('Starting training...')
    best_success_rate = 0.0
    
    for epoch in range(args.epochs):
        policy.train()
        total_loss = 0.0
        num_batches = 0
        
        # Training
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            optimizer.zero_grad()
            
            # Compute loss
            loss = compute_flow_matching_loss(policy, batch, device)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Evaluation
        if (epoch + 1) % args.eval_freq == 0:
            print(f'\nEvaluating at epoch {epoch+1}...')
            metrics = evaluate_policy(policy, env, num_episodes=args.eval_episodes, device=device)
            
            writer.add_scalar('Metrics/MeanReward', metrics['mean_reward'], epoch)
            writer.add_scalar('Metrics/SuccessRate', metrics['success_rate'], epoch)
            
            print(f'Mean Reward: {metrics["mean_reward"]:.3f}')
            print(f'Success Rate: {metrics["success_rate"]:.3f}')
            
            # Save best model
            if metrics['success_rate'] > best_success_rate:
                best_success_rate = metrics['success_rate']
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'success_rate': best_success_rate,
                    'args': args
                }, checkpoint_path)
                print(f'Saved best model to {checkpoint_path}')
        
        scheduler.step()
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FlowPolicy')
    
    # Environment args
    parser.add_argument('--env_name', type=str, default='FrankaKitchen-v1',
                       help='Environment name')
    parser.add_argument('--tasks', nargs='+', default=['microwave', 'kettle'],
                       help='Tasks to complete')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='Number of points in point cloud')
    parser.add_argument('--use_rgb', action='store_true',
                       help='Use RGB features in point cloud')
    
    # Data args
    parser.add_argument('--num_demos', type=int, default=50,
                       help='Number of expert demonstrations')
    parser.add_argument('--demo_path', type=str, default=None,
                       help='Path to saved demonstrations')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum steps per episode')
    
    # Model args
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension for point cloud encoder')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for flow matching network')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Evaluation args
    parser.add_argument('--eval_freq', type=int, default=10,
                       help='Evaluation frequency (epochs)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    
    # Logging args
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    train(args)
