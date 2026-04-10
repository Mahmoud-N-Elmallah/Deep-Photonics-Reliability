from utils import *
from data_pipeline import *
from model import *
from training_engine import train_model
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
    model = PhotonicResNet50(input_channels=input_channels, dropout_prob=config['model_hp']['drop_out']).to(device)
    
    # STAGE 1 SETUP: Freeze backbone, train head only
    staged_config = config.get('staged_finetuning', {})
    use_staged = staged_config.get('enabled', True)
    
    if use_staged:
        stage1_epochs = staged_config.get('stage1_epochs', 20)
        stage1_lr = staged_config.get('stage1_lr', 1e-3)
        stage2_lr = staged_config.get('stage2_lr', 1e-4)
        print(f"\n=== TWO-STAGE FINE-TUNING ===")
        print(f"Stage 1: Train head only for {stage1_epochs} epochs at LR={stage1_lr}")
        print(f"Stage 2: Fine-tune layer4+head for {config['model_hp']['epochs']-stage1_epochs} epochs at LR={stage2_lr}\n")
    else:
        unfreeze_head(model)
        stage1_epochs = 0
        stage1_lr = config['model_hp']['lr']
        stage2_lr = config['model_hp']['lr']
    
    # Resume from checkpoint if it exists
    if resume_checkpoint_path.exists():
        print(f"Resuming from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed at epoch {start_epoch}")
        
        # Detect which stage to resume in
        if use_staged and start_epoch >= stage1_epochs:
            print(f"Resuming in STAGE 2 (layer4 + head fine-tuning)")
            # Already unfrozen by checkpoint state
            unfreeze_layer4(model)
            unfreeze_head(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage2_lr,
                                        weight_decay=config['model_hp']['weight_decay'])
        else:
            print(f"Resuming in STAGE 1 (head-only training)")
            freeze_backbone(model)
            unfreeze_head(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage1_lr,
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
        print("Starting fresh training...")
        if use_staged:
            freeze_backbone(model)
            unfreeze_head(model)
        else:
            unfreeze_head(model)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage1_lr,
                                    weight_decay=config['model_hp']['weight_decay'])
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = CosineAnnealingLR(optimizer, T_max=stage1_epochs) if use_staged else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Get total epochs from config
    total_epochs = config['model_hp']['epochs']
    
    history, model = train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, total_epochs, device, history, checkpoint_dir, start_epoch, config, stage1_epochs if use_staged else None, stage2_lr if use_staged else None)
    
    # Save history after training
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()