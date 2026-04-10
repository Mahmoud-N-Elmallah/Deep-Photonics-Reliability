from utils import *
from data_pipeline import *
from model import *
from training_engine import *
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
import shutil
from pathlib import Path

def main():
    # Resolve paths mess
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    config_path = script_dir / 'config.yaml'
    config = load_config(str(config_path))
    device = setup_device()
    
    # Get experiment type from config
    experiment_type = config.get('experiment', {}).get('name', 'dual_channel')
    print(f"\nExperiment: {experiment_type}")
    
    # Compute class weights for loss function
    class_weights = compute_class_weights(config, device=device)
    
    # Create experiment-specific checkpoint directory
    checkpoint_dir = project_root / 'checkpoints' / experiment_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}\n")
    
    # Handle backward compatibility: migrate old checkpoints
    old_checkpoint_dir = project_root / 'checkpoints'
    old_latest = old_checkpoint_dir / 'latest_checkpoint.pkl'
    old_history = old_checkpoint_dir / 'training_history.pkl'
    
    if experiment_type == 'dual_channel' and old_latest.exists() and not (checkpoint_dir / 'latest_checkpoint.pkl').exists():
        print("Migrating old checkpoints to experiment-specific directory...")
        if old_latest.exists():
            shutil.copy(old_latest, checkpoint_dir / 'latest_checkpoint.pkl')
            print(f"  Migrated latest_checkpoint.pkl")
        if old_history.exists():
            shutil.copy(old_history, checkpoint_dir / 'training_history.pkl')
            print(f"  Migrated training_history.pkl")
        print()
    
    # Load existing history if it exists
    history_path = checkpoint_dir / 'training_history.pkl'
    if history_path.exists():
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    else:
        history = {}
    
    # Check if resuming from checkpoint or starting fresh
    resume_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pkl'
    start_epoch = 0
    
    train_loader, val_loader, test_loader, input_channels = build_loaders(config, project_root, experiment_type)
    num_classes = len(config.get('train_class_count', {0: 1, 1: 1, 2: 1, 3: 1}))
    model = PhotonicResNet18(input_channels=input_channels, num_classes=num_classes, dropout_prob=config['model_hp']['drop_out']).to(device)
    
    # Always ensure model is fully unfrozen
    for param in model.parameters():
        param.requires_grad = True
        
    # Resume from checkpoint if it exists
    if resume_checkpoint_path.exists():
        print(f"Resuming from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed at epoch {start_epoch}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['model_hp']['lr'],
                                    weight_decay=config['model_hp']['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Check if training already completed
        if start_epoch >= config['model_hp']['epochs']:
            print(f"\n  Training already completed {start_epoch} epochs (config specifies {config['model_hp']['epochs']} total).")
            print(f"To continue training, increase 'epochs' in config.yaml")
            print(f"Current: epochs: {config['model_hp']['epochs']}")
            print(f"Suggestion: epochs: {config['model_hp']['epochs'] + 20}  # to train 20 more epochs")
            print("\nNo new epochs will be trained. Exiting.")
            return
    else:
        print("Starting fresh training (fully unfrozen)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['model_hp']['lr'],
                                    weight_decay=config['model_hp']['weight_decay'])
    
    label_smoothing = config.get('model_hp', {}).get('label_smoothing', 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    # Use ReduceLROnPlateau for the entire training
    scheduler_factor = config.get('model_hp', {}).get('scheduler_factor', 0.5)
    scheduler_patience = config.get('model_hp', {}).get('scheduler_patience', 10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    
    # Get total epochs from config
    total_epochs = config['model_hp']['epochs']
    
    history, model = train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, total_epochs, device, history, checkpoint_dir, start_epoch, config)
    
    # Save history after training
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()