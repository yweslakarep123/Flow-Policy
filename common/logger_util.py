import heapq
import os
import logging
from termcolor import cprint


def setup_logger(log_dir, name='train'):
    """Setup logger"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class LargestKRecorder:
    def __init__(self, K):
        """
        Initialize the EfficientScalarRecorder.
        
        Parameters:
        - K: Number of largest scalars to consider when computing the average.
        """
        self.scalars = []
        self.K = K

    def record(self, scalar):
        """
        Record a scalar value.
        
        Parameters:
        - scalar: The scalar value to be recorded.
        """
        if len(self.scalars) < self.K:
            heapq.heappush(self.scalars, scalar)
        else:
            # Compare the new scalar with the smallest value in the heap
            if scalar > self.scalars[0]:
                heapq.heappushpop(self.scalars, scalar)

    def average_of_largest_K(self):
        """
        Compute the average of the largest K scalar values recorded.
        
        Returns:
        - avg: Average of the largest K scalars.
        """
        if len(self.scalars) == 0:
            raise ValueError("No scalars have been recorded yet.")
        
        return sum(self.scalars) / len(self.scalars)
