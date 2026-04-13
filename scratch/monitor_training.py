import torch
import os
import pickle
from pathlib import Path

def load_history_artifact(path):
    artifact_path = Path(path)
    if artifact_path.name == 'training_history.pkl':
        with open(artifact_path, 'rb') as f:
            return {'history': pickle.load(f)}
    payload = torch.load(artifact_path, map_location='cpu', weights_only=False)
    if 'history' in payload:
        return payload
    return {'history': payload}

def monitor():
    path = r'checkpoints/tri_channel/latest_checkpoint.pkl'
    fallback_path = r'checkpoints/tri_channel/training_history.pkl'
    if not os.path.exists(path):
        if not os.path.exists(fallback_path):
            print(f"Checkpoint not found at {path}")
            return
        path = fallback_path

    try:
        cp = load_history_artifact(path)
        h = cp['history']
        current_epoch = cp.get('epoch', len(h.get('train_loss', [])))
        best_f1 = cp.get('best_f1', h.get('best_val_f1', 0.0))
        print(f"Current Epoch: {current_epoch}")
        print(f"Best Val F1 so far: {best_f1:.4f}")
        print()
        
        # Determine number of epochs recorded
        n = len(h['train_loss'])
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train F1':>8} | {'Val Loss':>8} | {'Val F1':>8} | {'LR':>10}")
        print('-' * 65)
        
        start = max(0, n - 10)
        for i in range(start, n):
            print(f"{i+1:5d} | {h['train_loss'][i]:10.4f} | {h['train_f1'][i]:8.4f} | {h['val_loss'][i]:8.4f} | {h['val_f1'][i]:8.4f} | {h['lr'][i]:.2e}")
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    monitor()
