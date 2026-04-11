import torch
import torch.nn as nn
from pathlib import Path
import pickle
import shutil

from utils import load_config, setup_device, get_loss_function, get_optimizer, get_scheduler
from data_pipeline import build_loaders
from model import PhotonicResNet18
from training_engine import train_model

def main():
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config = load_config(str(script_dir / 'config.yaml'))
    device = setup_device()
    
    experiment_type = config.get('experiment', {}).get('name', 'tri_channel')
    checkpoint_dir = project_root / 'checkpoints' / experiment_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment: {experiment_type}")
    print(f"Checkpoint directory: {checkpoint_dir}\n")
    
    history_path = checkpoint_dir / 'training_history.pkl'
    history = {}
    if history_path.exists():
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
            
    resume_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pkl'
    start_epoch = 0
    
    train_loader, val_loader, test_loader, input_channels = build_loaders(config, project_root, experiment_type)
    num_classes = len(config.get('train_class_count', {0: 1, 1: 1, 2: 1, 3: 1}))
    model = PhotonicResNet18(
        input_channels=input_channels, 
        num_classes=num_classes, 
        dropout_prob=config['model_hp']['drop_out']
    ).to(device)
    
    for param in model.parameters():
        param.requires_grad = True
    optimizer = get_optimizer(model, config)
    loss_fn = get_loss_function(config, device)
    scheduler = get_scheduler(optimizer, config)
    
    # Resume Logic
    if resume_checkpoint_path.exists():
        print(f"Resuming from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed at epoch {start_epoch}")
        
        if start_epoch >= config['model_hp']['epochs']:
            print("Training already complete.")
            return

  
    history, model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        epochs=config['model_hp']['epochs'],
        device=device,
        history=history,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        config=config
    )
    

    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()