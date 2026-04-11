import os
import torch
import torch.nn as nn
from pathlib import Path
import yaml

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
    checkpoint_dir = project_root / 'checkpoints' / 'phase4_physics'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    
    print("Building Physics-Aware DataLoaders...")
    train_loader, val_loader, test_loader, input_channels = build_loaders(
        config, project_root, experiment_type, is_physics=True
    )
    
    # Initialize Model and Load Weights from Phase 3
    num_classes = len(config.get('train_class_count', {0: 1, 1: 1, 2: 1, 3: 1}))
    model = PhotonicResNet18(
        input_channels=input_channels, 
        num_classes=num_classes,
        dropout_prob=config.get('model_hp', {}).get('drop_out', 0.5)
    ).to(device)
    
    # Load Best Model from Phase 3 as the starting point for Phase 4
    phase3_checkpoint = project_root / 'checkpoints' / experiment_type / 'best_model.pth'
    if phase3_checkpoint.exists():
        print(f"Loading Phase 3 Pretrained Weights: {phase3_checkpoint}")
        model.load_state_dict(torch.load(str(phase3_checkpoint), map_location=device))
    else:
        print("WARNING: Phase 3 checkpoint not found. Starting from scratch/ImageNet.")

    
    optimizer = get_optimizer(model, config)
    loss_fn = get_loss_function(config, device)
    scheduler = get_scheduler(optimizer, config)
    
    epochs = config.get('model_hp', {}).get('physics_epochs', 30) 
    # Phase 4 Training
    print(f"\nStarting Phase 4: Physics-Constrained Supervision")
    print(f"Physics Lambda: {config.get('model_hp', {}).get('physics_lambda', 1.0)}")
    
    history, trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        checkpoint_dir=checkpoint_dir,
        config=config,
        is_physics=True
    )
    
    # Save History
    import pickle
    history_path = checkpoint_dir / 'training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\nPhase 4 Training Complete!")
    print(f"Final Models and History saved in: {checkpoint_dir}")

if __name__ == '__main__':
    main()
