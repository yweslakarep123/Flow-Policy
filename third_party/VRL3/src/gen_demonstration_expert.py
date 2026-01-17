# usage:
#       bash scripts/vrl3_gen_demonstration_expert.sh door
import mj_envs
from mjrl.utils.gym_env import GymEnv
from rrl_local.rrl_utils import make_basic_env, make_dir
from adroit import AdroitEnv
import matplotlib.pyplot as plt
import argparse
import os
import torch
from vrl3_agent import VRL3Agent
import utils
from termcolor import cprint
from PIL import Image
import zarr
from copy import deepcopy
import numpy as np
import sys
import traceback

# Simple and direct - sesuaikan dengan struktur yang benar
sys.path.insert(0, "/home/dapa/Documents/FlowPolicy")
sys.path.insert(0, "/home/dapa/Documents/FlowPolicy/FlowPolicy")

try:
    from flow_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit
    print("✅ Successfully imported MujocoPointcloudWrapperAdroit")
except ImportError as e:
    print(f"❌ Error importing MujocoPointcloudWrapperAdroit: {e}")
    print("Trying alternative import paths...")
    # Coba path alternatif
    try:
        from gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit
        print("✅ Successfully imported from gym_util.mjpc_wrapper")
    except ImportError:
        print("❌ All import attempts failed")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='door', help='environment to run')
    parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to run')
    parser.add_argument('--root_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--expert_ckpt_path', type=str, default=None, help='path to expert ckpt')
    parser.add_argument('--img_size', type=int, default=84, help='image size')
    parser.add_argument('--not_use_multi_view', action='store_true', help='not use multi view')
    parser.add_argument('--use_point_crop', action='store_true', help='use point crop')
    args = parser.parse_args()
    return args

def fix_paths(args):
    """Fix paths from /data/ to local home directory"""
    import os
    
    # Get home directory
    home = os.path.expanduser("~")
    current_dir = os.getcwd()
    
    print(f"Home directory: {home}")
    print(f"Current directory: {current_dir}")
    
    # Debug: Show original paths
    print(f"Original root_dir: {args.root_dir}")
    print(f"Original expert_ckpt_path: {args.expert_ckpt_path}")
    
    # Jika root_dir mengandung /data/, ubah ke home directory
    if args.root_dir and args.root_dir.startswith('/data/'):
        # Coba beberapa kemungkinan mapping
        if '/data/code/FlowPolicy/' in args.root_dir:
            # Mapping 1: /data/code/FlowPolicy/ -> /home/dapa/Documents/FlowPolicy/
            args.root_dir = args.root_dir.replace('/data/code/FlowPolicy/', f'{home}/Documents/FlowPolicy/')
        else:
            # Mapping umum: /data/ -> /home/dapa/data/
            args.root_dir = args.root_dir.replace('/data/', f'{home}/data/')
        print(f"Fixed root_dir to: {args.root_dir}")
    
    # Jika expert_ckpt_path mengandung /data/, ubah juga
    if args.expert_ckpt_path and args.expert_ckpt_path.startswith('/data/'):
        if '/data/code/FlowPolicy/' in args.expert_ckpt_path:
            # Checkpoint ada di /home/dapa/Documents/FlowPolicy/third_party/
            args.expert_ckpt_path = args.expert_ckpt_path.replace('/data/code/FlowPolicy/', f'{home}/Documents/FlowPolicy/')
        else:
            args.expert_ckpt_path = args.expert_ckpt_path.replace('/data/', f'{home}/data/')
        print(f"Fixed expert_ckpt_path to: {args.expert_ckpt_path}")
    
    # Buat direktori jika belum ada
    if args.root_dir:
        os.makedirs(args.root_dir, exist_ok=True)
        print(f"Created/verified directory: {args.root_dir}")
    
    return args

def find_checkpoint(env_name, expert_ckpt_path):
    """Find checkpoint file with multiple fallback options"""
    import os
    
    # Daftar semua lokasi yang mungkin
    possible_paths = [
        # 1. Path yang diberikan
        expert_ckpt_path,
        
        # 2. Lokasi sebenarnya berdasarkan struktur Anda
        f"/home/dapa/Documents/FlowPolicy/third_party/VRL3/vrl3_ckpts/vrl3_{env_name}.pt",
        
        # 3. Lokasi relatif dari script ini
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                    f"../../vrl3_ckpts/vrl3_{env_name}.pt"),
        
        # 4. Lokasi di current working directory
        os.path.join(os.getcwd(), f"third_party/VRL3/vrl3_ckpts/vrl3_{env_name}.pt"),
        
        # 5. Lokasi dengan duplikat FlowPolicy
        f"/home/dapa/Documents/FlowPolicy/FlowPolicy/third_party/VRL3/vrl3_ckpts/vrl3_{env_name}.pt",
        
        # 6. Coba tanpa 'vrl3_' prefix
        f"/home/dapa/Documents/FlowPolicy/third_party/VRL3/vrl3_ckpts/{env_name}.pt",
        
        # 7. Coba di parent directory
        f"/home/dapa/Documents/FlowPolicy/vrl3_ckpts/vrl3_{env_name}.pt",
    ]
    
    # Tambahkan variasi lain
    home = os.path.expanduser("~")
    possible_paths.extend([
        f"{home}/Documents/FlowPolicy/third_party/VRL3/vrl3_ckpts/vrl3_{env_name}.pt",
        f"{home}/FlowPolicy/third_party/VRL3/vrl3_ckpts/vrl3_{env_name}.pt",
    ])
    
    # Cek setiap path
    for i, path in enumerate(possible_paths):
        if path and os.path.exists(path):
            print(f"✅ Found checkpoint at option {i+1}: {path}")
            return os.path.abspath(path)
        elif path:
            print(f"  Option {i+1}: {path} - Not found")
    
    return None

def render_camera(sim, camera_name="top"):
    img = sim.render(84, 84, camera_name=camera_name)
    return img

def render_high_res(sim, camera_name="top"):
    img = sim.render(1024, 1024, camera_name=camera_name)
    return img

def main():
    try:
        args = parse_args()
        
        print("=" * 60)
        print(f"Starting demonstration generation for: {args.env_name}")
        print("=" * 60)
        
        # Fix paths to avoid permission issues
        args = fix_paths(args)
        
        # Temukan checkpoint
        print("\n" + "=" * 60)
        print("Looking for checkpoint file...")
        print("=" * 60)
        
        checkpoint_path = find_checkpoint(args.env_name, args.expert_ckpt_path)
        
        if not checkpoint_path:
            cprint(f"❌ ERROR: Could not find checkpoint for {args.env_name}", 'red')
            cprint("Please ensure the checkpoint file exists.", 'red')
            cprint("You might need to download it first or check the path.", 'red')
            
            # Coba list files di directory checkpoint
            ckpt_dir = "/home/dapa/Documents/FlowPolicy/third_party/VRL3/vrl3_ckpts/"
            if os.path.exists(ckpt_dir):
                cprint(f"Files in {ckpt_dir}:", 'yellow')
                files = os.listdir(ckpt_dir)
                for f in files:
                    print(f"  - {f}")
            return
        
        # Update args dengan checkpoint path yang ditemukan
        args.expert_ckpt_path = checkpoint_path
        print(f"✅ Using checkpoint: {checkpoint_path}")
        
        # load env
        action_repeat = 2
        frame_stack = 1
        
        def create_env():
            try:
                print(f"Creating environment for {args.env_name}-v0")
                env = AdroitEnv(
                    env_name=args.env_name+'-v0', 
                    test_image=False, 
                    num_repeats=action_repeat,
                    num_frames=frame_stack, 
                    env_feature_type='pixels',
                    device='cuda', 
                    reward_rescale=True
                )
                print(f"Wrapping environment with MujocoPointcloudWrapperAdroit")
                env = MujocoPointcloudWrapperAdroit(
                    env=env, 
                    env_name='adroit_'+args.env_name, 
                    use_point_crop=args.use_point_crop
                )
                print("✅ Environment created successfully")
                return env
            except Exception as e:
                cprint(f"❌ Error creating environment: {e}", 'red')
                traceback.print_exc()
                return None
        
        num_episodes = args.num_episodes
        save_dir = os.path.join(args.root_dir, f'adroit_{args.env_name}_expert.zarr')
        
        print(f"\nSave directory: {save_dir}")
        
        # Create parent directory for save_dir
        parent_dir = os.path.dirname(save_dir) if os.path.dirname(save_dir) else '.'
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Created parent directory: {parent_dir}")
        
        # Check if output already exists
        if os.path.exists(save_dir):
            cprint(f'⚠️  Data already exists at {save_dir}', 'yellow')
            cprint("If you want to overwrite, delete the existing directory first.", "yellow")
            cprint("Do you want to overwrite? (y/n)", "yellow")
            user_input = 'y'  # Auto-yes for batch processing
            # user_input = input().strip().lower()
            
            if user_input == 'y':
                cprint(f'🗑️  Overwriting {save_dir}', 'red')
                import shutil
                if os.path.isdir(save_dir):
                    shutil.rmtree(save_dir)
                else:
                    os.remove(save_dir)
            else:
                cprint('👋 Exiting', 'red')
                return
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        print(f"✅ Created save directory: {save_dir}")
        
        # load expert ckpt
        print(f"\nLoading checkpoint: {checkpoint_path}")
        
        try:
            # Check file size first
            file_size = os.path.getsize(checkpoint_path)
            print(f"Checkpoint file size: {file_size / (1024*1024):.2f} MB")
            
            # Load the checkpoint
            loaded_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Check what's in the checkpoint
            print(f"Checkpoint keys: {list(loaded_dict.keys())}")
            
            # Extract agent
            if 'agent' in loaded_dict:
                expert_agent = loaded_dict['agent']
                print("✅ Found 'agent' in checkpoint")
            elif 'model' in loaded_dict:
                expert_agent = loaded_dict['model']
                print("✅ Found 'model' in checkpoint")
            elif 'state_dict' in loaded_dict:
                # Create agent first, then load state dict
                print("⚠️  Checkpoint contains 'state_dict', need to create agent first")
                # This part depends on your VRL3Agent implementation
                expert_agent = VRL3Agent()  # You might need to adjust this
                expert_agent.load_state_dict(loaded_dict['state_dict'])
            else:
                cprint("❌ ERROR: Checkpoint doesn't contain 'agent', 'model', or 'state_dict'", 'red')
                return
            
            # PERBAIKAN: VRL3Agent bukan torch.nn.Module, gunakan metode .to() dari kelas VRL3Agent
            if torch.cuda.is_available():
                print("CUDA is available")
                # VRL3Agent memiliki metode .to(device) sendiri
                expert_agent.to('cuda')
                print("✅ Moved agent to CUDA using expert_agent.to('cuda')")
            else:
                print("⚠️  CUDA not available, using CPU")
            
            # Set to eval mode jika ada
            if hasattr(expert_agent, 'eval'):
                expert_agent.eval()
                print("✅ Set agent to eval mode")
            elif hasattr(expert_agent, 'train'):
                # Jika ada mode train, set ke False
                expert_agent.train(False)
                print("✅ Set agent to evaluation mode")
            
            cprint(f'✅ Successfully loaded expert checkpoint', 'green')
            
        except Exception as e:
            cprint(f'❌ Failed to load checkpoint: {e}', 'red')
            traceback.print_exc()
            return

        # Initialize data storage
        total_count = 0
        img_arrays = []
        point_cloud_arrays = []
        depth_arrays = []
        state_arrays = []
        action_arrays = []
        episode_ends_arrays = []
        
        print(f"\n" + "=" * 60)
        print(f"Starting data collection for {num_episodes} episodes")
        print("=" * 60)
        
        # loop over episodes
        minimal_episode_length = 100
        episode_idx = 0
        
        while episode_idx < num_episodes:
            print(f"\n--- Episode {episode_idx + 1}/{num_episodes} ---")
            
            # Create environment for this episode
            env = create_env()
            if env is None:
                cprint("❌ Failed to create environment", 'red')
                break
            
            try:
                time_step = env.reset()
            except Exception as e:
                cprint(f"❌ Error resetting environment: {e}", 'red')
                del env
                continue
            
            input_obs_visual = time_step.observation  # (3n,84,84), unit8
            input_obs_sensor = time_step.observation_sensor  # float32
            
            total_reward = 0.
            n_goal_achieved_total = 0.
            step_count = 0
            
            img_arrays_sub = []
            point_cloud_arrays_sub = []
            depth_arrays_sub = []
            state_arrays_sub = []
            action_arrays_sub = []
            total_count_sub = 0
            
            try:
                while (not time_step.last()) or step_count < minimal_episode_length:
                    with torch.no_grad():
                        # Gunakan utils.eval_mode hanya jika agent memiliki training mode
                        if hasattr(expert_agent, 'train'):
                            with utils.eval_mode(expert_agent):
                                input_obs_visual = time_step.observation
                                input_obs_sensor = time_step.observation_sensor
                                
                                # Get action from expert agent
                                action = expert_agent.act(
                                    obs=input_obs_visual, 
                                    step=0,
                                    eval_mode=True, 
                                    obs_sensor=input_obs_sensor
                                )  # (28,) float32
                        else:
                            # Jika agent tidak memiliki mode training
                            input_obs_visual = time_step.observation
                            input_obs_sensor = time_step.observation_sensor
                            
                            # Get action from expert agent
                            action = expert_agent.act(
                                obs=input_obs_visual, 
                                step=0,
                                eval_mode=True, 
                                obs_sensor=input_obs_sensor
                            )  # (28,) float32
                        
                        if args.not_use_multi_view:
                            input_obs_visual = input_obs_visual[:3]  # (3,84,84)
                        
                        # save data
                        total_count_sub += 1
                        img_arrays_sub.append(input_obs_visual.copy())
                        state_arrays_sub.append(input_obs_sensor.copy())
                        action_arrays_sub.append(action.copy())
                        point_cloud_arrays_sub.append(time_step.observation_pointcloud.copy())
                        depth_arrays_sub.append(time_step.observation_depth.copy())
                    
                    time_step = env.step(action)
                    obs = time_step.observation  # np array, (3,84,84)
                    obs = obs[:3] if obs.shape[0] > 3 else obs  # (3,84,84)
                    n_goal_achieved_total += time_step.n_goal_achieved
                    total_reward += time_step.reward
                    step_count += 1
                    
                    # Progress indicator
                    if step_count % 50 == 0:
                        print(f"  Step {step_count}, Reward: {total_reward:.2f}")
                
            except Exception as e:
                cprint(f"❌ Error during episode execution: {e}", 'red')
                traceback.print_exc()
                del env
                continue
            
            # Episode quality check
            if n_goal_achieved_total < 10.:
                cprint(f"  Episode {episode_idx} discarded - only {n_goal_achieved_total} goals achieved", 'yellow')
                del env
                continue
            
            # Store successful episode data
            total_count += total_count_sub
            episode_ends_arrays.append(deepcopy(total_count))
            img_arrays.extend(deepcopy(img_arrays_sub))
            point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
            depth_arrays.extend(deepcopy(depth_arrays_sub))
            state_arrays.extend(deepcopy(state_arrays_sub))
            action_arrays.extend(deepcopy(action_arrays_sub))
            
            print(f"  ✅ Episode {episode_idx}: {step_count} steps, "
                  f"Reward: {total_reward:.2f}, "
                  f"Goals: {n_goal_achieved_total}")
            
            episode_idx += 1
            del env
        
        print(f"\n" + "=" * 60)
        print(f"Data collection complete. Collected {episode_idx} episodes")
        print(f"Total steps: {total_count}")
        print("=" * 60)
        
        if episode_idx == 0:
            cprint("❌ No episodes were successfully collected", 'red')
            return
        
        ###############################
        # save data
        ###############################
        print(f"\nSaving data to {save_dir}")
        
        try:
            # Convert lists to numpy arrays
            img_arrays = np.stack(img_arrays, axis=0)
            if img_arrays.shape[1] == 3:  # make channel last
                img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
            state_arrays = np.stack(state_arrays, axis=0)
            point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
            depth_arrays = np.stack(depth_arrays, axis=0)
            action_arrays = np.stack(action_arrays, axis=0)
            episode_ends_arrays = np.array(episode_ends_arrays)
            
            # Print shapes for verification
            print(f"  Image array shape: {img_arrays.shape}")
            print(f"  State array shape: {state_arrays.shape}")
            print(f"  Point cloud array shape: {point_cloud_arrays.shape}")
            print(f"  Depth array shape: {depth_arrays.shape}")
            print(f"  Action array shape: {action_arrays.shape}")
            print(f"  Episode ends shape: {episode_ends_arrays.shape}")
            
            # Create zarr file
            zarr_root = zarr.group(save_dir)
            zarr_data = zarr_root.create_group('data')
            zarr_meta = zarr_root.create_group('meta')
            
            # Configure compression
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            
            # Set chunk sizes
            img_chunk_size = (min(100, len(img_arrays)), img_arrays.shape[1], 
                            img_arrays.shape[2], img_arrays.shape[3])
            state_chunk_size = (min(100, len(state_arrays)), state_arrays.shape[1])
            point_cloud_chunk_size = (min(100, len(point_cloud_arrays)), 
                                    point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
            depth_chunk_size = (min(100, len(depth_arrays)), 
                              depth_arrays.shape[1], depth_arrays.shape[2])
            action_chunk_size = (min(100, len(action_arrays)), action_arrays.shape[1])
            
            # Create datasets
            zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, 
                                    dtype='uint8', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, 
                                    dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, 
                                    chunks=point_cloud_chunk_size, dtype='float32', 
                                    overwrite=True, compressor=compressor)
            zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, 
                                    dtype='float32', overwrite=True, compressor=compressor)
            zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, 
                                    dtype='float32', overwrite=True, compressor=compressor)
            zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, 
                                    dtype='int64', overwrite=True, compressor=compressor)
            
            # Print success message with statistics
            cprint('\n' + '=' * 60, 'green')
            cprint('✅ DATA SAVED SUCCESSFULLY', 'green')
            cprint('=' * 60, 'green')
            cprint(f'Environment: {args.env_name}', 'green')
            cprint(f'Episodes collected: {episode_idx}', 'green')
            cprint(f'Total steps: {total_count}', 'green')
            cprint(f'Saved to: {save_dir}', 'green')
            cprint('=' * 60, 'green')
            
            # Print detailed statistics
            print(f"\n📊 Dataset Statistics:")
            print(f"  Images: {img_arrays.shape}, range: [{img_arrays.min():.2f}, {img_arrays.max():.2f}]")
            print(f"  States: {state_arrays.shape}, range: [{state_arrays.min():.2f}, {state_arrays.max():.2f}]")
            print(f"  Point clouds: {point_cloud_arrays.shape}, range: [{point_cloud_arrays.min():.2f}, {point_cloud_arrays.max():.2f}]")
            print(f"  Depth: {depth_arrays.shape}, range: [{depth_arrays.min():.2f}, {depth_arrays.max():.2f}]")
            print(f"  Actions: {action_arrays.shape}, range: [{action_arrays.min():.2f}, {action_arrays.max():.2f}]")
            
        except Exception as e:
            cprint(f'❌ Error saving data: {e}', 'red')
            traceback.print_exc()
            
    except KeyboardInterrupt:
        cprint('\n⚠️  Process interrupted by user', 'yellow')
    except Exception as e:
        cprint(f'\n❌ Unexpected error: {e}', 'red')
        traceback.print_exc()
    finally:
        print("\n" + "=" * 60)
        print("Process completed")
        print("=" * 60)

if __name__ == '__main__':
    main()