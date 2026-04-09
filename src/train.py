from src.utils import *
from src.data_pipeline import *
from src.model import *
from src.training_engine import train_model
import torch.nn as nn
import torch
import pickle
from pathlib import Path

def main():
    config = load_config('../src/config.yaml')
    device = setup_device()
    
    # Inject computed weights into config
    config['class_weight'] = compute_class_weights(config)
    
    # Load existing history if it exists
    history_path = Path('training_history.pkl')
    if history_path.exists():
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    else:
        history = {}
    
    train_loader, val_loader, test_loader = build_loaders(config)
    model = PhotonicResNet50().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model_hp']['lr'],
                                  weight_decay=config['model_hp']['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    history, model = train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, config['model_hp']['epochs'], device, history)
    
    # Save history after training
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()