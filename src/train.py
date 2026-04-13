import argparse
import torch
from pathlib import Path

from utils import (
    add_config_argument,
    get_loss_function,
    get_optimizer,
    get_scheduler,
    load_pickle,
    load_runtime_config,
    save_pickle,
    setup_device,
)
from data_pipeline import build_loaders
from model import PhotonicResNet18
from training_engine import train_model

def main():
    parser = argparse.ArgumentParser(description="Train baseline Phase 1-2 model.")
    add_config_argument(parser, default="src/config.yaml")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config, config_path = load_runtime_config(args.config, project_root)
    device = setup_device()
    
    experiment_type = config.get('experiment', {}).get('name', 'tri_channel')
    checkpoint_dir = project_root / 'checkpoints' / experiment_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment: {experiment_type}")
    print(f"Checkpoint directory: {checkpoint_dir}\n")
    
    history_path = checkpoint_dir / 'training_history.pkl'
    resume_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pkl'
    history = load_pickle(history_path) if history_path.exists() and resume_checkpoint_path.exists() else {}
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
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('history'):
            history = checkpoint['history']
        start_epoch = checkpoint.get('epoch', 0)
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
    

    save_pickle(history, history_path)
    print(f"Training history saved to {history_path}")
    print(f"Config used: {config_path}")

if __name__ == "__main__":
    main()
