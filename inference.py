"""
Inference script untuk FlowPolicy
"""
import argparse
import torch
import numpy as np
from tqdm import tqdm
import os

from policies.flow_policy import FlowPolicy
from envs.franka_kitchen_wrapper import make_franka_kitchen_env
from utils.training_utils import evaluate_policy


def inference(args):
    """Main inference function"""
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
    
    # Load model
    print(f'Loading model from {args.checkpoint_path}...')
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Create model
    policy = FlowPolicy(
        action_dim=action_dim,
        point_cloud_dim=3,
        feature_dim=args.feature_dim,
        num_points=args.num_points,
        hidden_dim=args.hidden_dim,
        use_rgb=args.use_rgb
    ).to(device)
    
    # Load weights
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f'Model loaded from epoch {checkpoint.get("epoch", "unknown")}')
    print(f'Success rate: {checkpoint.get("success_rate", "unknown")}')
    
    # Evaluation
    print(f'\nEvaluating policy on {args.num_episodes} episodes...')
    metrics = evaluate_policy(policy, env, num_episodes=args.num_episodes, device=device)
    
    print('\n=== Evaluation Results ===')
    print(f'Mean Reward: {metrics["mean_reward"]:.3f} ± {metrics["std_reward"]:.3f}')
    print(f'Success Rate: {metrics["success_rate"]:.3f}')
    
    # Visualize episodes if requested
    if args.render:
        print('\nRendering episodes...')
        render_episode(policy, env, device, args.num_points, args.use_rgb, args.num_steps)


def render_episode(policy, env, device, num_points, use_rgb, num_steps=1):
    """Render a single episode"""
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0
    max_steps = 500
    
    while not done and step < max_steps:
        # Render environment
        env.render()
        
        # Get point cloud
        point_cloud = torch.FloatTensor(obs['point_cloud']).unsqueeze(0).to(device)
        rgb_features = None
        if use_rgb and 'rgb_features' in obs:
            rgb_features = torch.FloatTensor(obs['rgb_features']).unsqueeze(0).to(device)
        
        # Predict action
        with torch.no_grad():
            action = policy.predict(point_cloud, rgb_features, num_steps=num_steps)
            action = action.cpu().numpy()[0]
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        print(f'Step {step}: Reward = {reward:.3f}, Total = {total_reward:.3f}')
    
    print(f'\nEpisode completed: Total reward = {total_reward:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FlowPolicy Inference')
    
    # Environment args
    parser.add_argument('--env_name', type=str, default='FrankaKitchen-v1',
                       help='Environment name')
    parser.add_argument('--tasks', nargs='+', default=['microwave', 'kettle'],
                       help='Tasks to complete')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='Number of points in point cloud')
    parser.add_argument('--use_rgb', action='store_true',
                       help='Use RGB features in point cloud')
    
    # Model args
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension for point cloud encoder')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for flow matching network')
    
    # Inference args
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--num_steps', type=int, default=1,
                       help='Number of inference steps (1 for single-step)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes visually')
    
    args = parser.parse_args()
    inference(args)
