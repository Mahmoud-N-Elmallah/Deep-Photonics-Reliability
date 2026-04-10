import torch
import os

def monitor():
    path = r'checkpoints/tri_channel/latest_checkpoint.pkl'
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return

    try:
        cp = torch.load(path, map_location='cpu', weights_only=False)
        h = cp['history']
        print(f"Current Epoch: {cp['epoch']}")
        print(f"Best Val F1 so far: {cp.get('best_f1', 0.0):.4f}")
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
