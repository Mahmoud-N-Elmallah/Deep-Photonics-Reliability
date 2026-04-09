from src.utils import *
from src.data_pipeline import *
from src.model import *
from src.training_engine import train_model
import torch.nn as nn
import torch

def main():
    config = load_config('src/config.yaml')
    device = setup_device()
    
    # Inject computed weights into config
    config['class_weight'] = compute_class_weights(config)
    
    train_loader, val_loader,test_loader = build_loaders(config)
    model = PhotonicResNet50().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model_hp']['lr'],
                                  weight_decay=config['model_hp']['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, config['epochs'], device)

if __name__ == "__main__":
    main()