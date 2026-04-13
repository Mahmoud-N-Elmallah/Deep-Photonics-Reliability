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
    parser = argparse.ArgumentParser(description="Run Phase 4 physics-aware fine-tuning.")
    add_config_argument(parser, default="src/config.yaml")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config, config_path = load_runtime_config(args.config, project_root)
    device = setup_device()
    
    experiment_type = config.get('experiment', {}).get('name', 'tri_channel')
    checkpoint_dir = project_root / 'checkpoints' / 'phase4_physics'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    
    print("Building Physics-Aware DataLoaders...")
    train_loader, val_loader, test_loader, input_channels = build_loaders(
        config, project_root, experiment_type, is_physics=True
    )
    print(f"Physics train/val samples: {len(train_loader.dataset)} / {len(val_loader.dataset)}")
    
    # Initialize Model and Load Weights from Phase 3
    num_classes = len(config.get('train_class_count', {0: 1, 1: 1, 2: 1, 3: 1}))
    model = PhotonicResNet18(
        input_channels=input_channels, 
        num_classes=num_classes,
        dropout_prob=config.get('model_hp', {}).get('drop_out', 0.5)
    ).to(device)
    
    # Load Best Model from Phase 3 as the starting point for Phase 4
    phase3_checkpoint = project_root / 'checkpoints' / experiment_type / 'best_model.pth'
    resume_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pkl'
    history_path = checkpoint_dir / 'training_history.pkl'
    history = load_pickle(history_path) if history_path.exists() and resume_checkpoint_path.exists() else {}

    optimizer = get_optimizer(model, config)
    loss_fn = get_loss_function(config, device)
    scheduler = get_scheduler(optimizer, config)
    start_epoch = 0

    if resume_checkpoint_path.exists():
        print(f"Resuming Phase 4 from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('history'):
            history = checkpoint['history']
        start_epoch = checkpoint.get('epoch', 0)
    else:
        if not phase3_checkpoint.exists():
            raise FileNotFoundError(
                f"Phase 3 checkpoint missing at {phase3_checkpoint}. Run Phase 1-3 before Phase 4."
            )
        print(f"Loading Phase 3 Pretrained Weights: {phase3_checkpoint}")
        model.load_state_dict(torch.load(str(phase3_checkpoint), map_location=device))
    
    epochs = config.get('model_hp', {}).get('physics_epochs', 30) 
    if start_epoch >= epochs:
        print("Phase 4 training already complete.")
        return

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
        history=history,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        config=config,
        is_physics=True
    )
    
    # Save History
    save_pickle(history, history_path)
    
    print(f"\nPhase 4 Training Complete!")
    print(f"Final Models and History saved in: {checkpoint_dir}")
    print(f"Config used: {config_path}")

if __name__ == '__main__':
    main()
