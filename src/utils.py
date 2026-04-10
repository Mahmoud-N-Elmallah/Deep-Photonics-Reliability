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
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize to sum to num_classes
    return class_weights.to(device)

def freeze_backbone(model) -> None:
    """Freeze backbone layers except conv1 and bn1 (re-initialized for custom input channels)."""
    for name, param in model.model.named_parameters():
        # Keep the top-level conv1/bn1 trainable since they were re-initialized
        # Note: layer1.0.conv1 etc. are NOT matched here (they contain a dot before conv1)
        if name.startswith('conv1.') or name.startswith('bn1.'):
            continue
        if name.startswith('fc.'):
            continue  # fc is handled separately by unfreeze_head
        param.requires_grad = False
    logger.info("Backbone frozen (conv1 and bn1 remain trainable for custom input channels).")

def unfreeze_layer4(model) -> None:
    """Unfreeze layer4 for stage 2 fine-tuning."""
    for param in model.model.layer4.parameters():
        param.requires_grad = True
    logger.info("Layer4 unfrozen for stage 2 fine-tuning.")

def unfreeze_head(model) -> None:
    """Ensure head (fc) is unfrozen for training."""
    for param in model.model.fc.parameters():
        param.requires_grad = True
    logger.info("Head unfrozen and ready for training.")