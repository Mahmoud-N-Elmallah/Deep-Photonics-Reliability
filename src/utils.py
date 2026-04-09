import logging
import torch
import yaml
from pathlib import Path
from typing import Dict

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

logger = setup_logger()

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def compute_class_weights(config: Dict) -> list:
    counts_dict = config['train_class_count']
    counts = torch.tensor([counts_dict[i] for i in range(len(counts_dict))], dtype=torch.float32)
    return (1.0 / counts).tolist()