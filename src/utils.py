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

def compute_class_weights(config: Dict, device: torch.device = None) -> torch.Tensor:
    """Compute class weights for CrossEntropyLoss (inverse frequency normalized)."""
    if device is None:
        device = torch.device('cpu')
    counts_dict = config['train_class_count']
    counts = torch.tensor([counts_dict[i] for i in range(len(counts_dict))], dtype=torch.float32)
    class_weights = 1.0 / torch.sqrt(counts)
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize to sum to num_classes
    return class_weights.to(device)

def get_loss_function(config: Dict, device: torch.device):
    """Factory for selecting loss function based on config."""
    class_weights = compute_class_weights(config, device=device)
    loss_type = config['model_hp'].get('loss_type', 'ce')
    label_smoothing = config['model_hp'].get('label_smoothing', 0.0)
    
    if loss_type == 'ce':
        return torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        # Fallback to standard CE
        return torch.nn.CrossEntropyLoss(weight=class_weights)

def get_optimizer(model, config: Dict):
    """Initialize AdamW optimizer with weight decay."""
    return torch.optim.AdamW(
        model.parameters(), 
        lr=config['model_hp']['lr'],
        weight_decay=config['model_hp']['weight_decay']
    )

def get_scheduler(optimizer, config: Dict):
    """Initialize ReduceLROnPlateau scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['model_hp'].get('scheduler_factor', 0.5), 
        patience=config['model_hp'].get('scheduler_patience', 10)
    )