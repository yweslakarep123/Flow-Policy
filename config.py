"""
Configuration file untuk FlowPolicy training dan inference
"""
import argparse


class Config:
    """Configuration class untuk FlowPolicy"""
    
    # Environment settings
    ENV_NAME = 'FrankaKitchen-v1'
    TASKS = ['microwave', 'kettle']
    NUM_POINTS = 1024
    USE_RGB = True
    
    # Model settings
    ACTION_DIM = 9  # Franka: 7 arm + 2 gripper
    POINT_CLOUD_DIM = 3
    FEATURE_DIM = 256
    HIDDEN_DIM = 512
    
    # Training settings
    NUM_DEMOS = 50
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    MAX_STEPS = 500
    
    # Evaluation settings
    EVAL_FREQ = 10
    EVAL_EPISODES = 10
    
    # Logging settings
    LOG_DIR = 'logs'
    CHECKPOINT_DIR = 'checkpoints'
    
    # Inference settings
    NUM_STEPS = 1  # 1 untuk single-step inference
    
    @classmethod
    def from_args(cls, args):
        """Create config from argparse args"""
        config = cls()
        for key, value in vars(args).items():
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)
        return config


def get_config():
    """Get default configuration"""
    return Config()
