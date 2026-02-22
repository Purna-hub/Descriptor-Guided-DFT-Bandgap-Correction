import os
import numpy as np

def set_seed(seed):
    np.random.seed(seed)

def create_directories():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
