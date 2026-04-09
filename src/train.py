from utils import *
from data_pipeline import *
from model import *
from training_engine import train_model
import torch.nn as nn
import torch
import pickle
from pathlib import Path

def main():
    # Resolve paths mess
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    config_path = script_dir / 'config.yaml'
    config = load_config(str(config_path))
    device = setup_device()
    
    # add computed weights into config
    config['class_weight'] = compute_class_weights(config)
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = project_root / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
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
    
    train_loader, val_loader, test_loader = build_loaders(config, project_root)
    model = PhotonicResNet50(dropout_prob=config['model_hp']['drop_out']).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model_hp']['lr2'],
                                  weight_decay=config['model_hp']['weight_decay2'])
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Resume from checkpoint if it exists
    if resume_checkpoint_path.exists():
        print(f"Resuming from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed at epoch {start_epoch}")
    else:
        print("Starting fresh training...")
    
    total_epochs = config['model_hp']['epochs']
    history, model = train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, total_epochs, device, history, checkpoint_dir, start_epoch)
    
    # Save history after training
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")
    
    # Save history after training
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()