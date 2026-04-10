import torch

cp = torch.load('checkpoints/dual_channel/latest_checkpoint.pkl', map_location='cpu', weights_only=False)
h = cp['history']
print(f"Training at epoch: {cp['epoch']}")
print(f"Best Val F1: {cp['best_f1']:.4f}")
print()
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train F1':>8} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val F1':>8} | {'Val Acc':>8} | {'LR':>10}")
print('-' * 95)
for i in range(len(h['train_loss'])):
    print(f"{i+1:5d} | {h['train_loss'][i]:10.4f} | {h['train_f1'][i]:8.4f} | {h['train_acc'][i]:9.4f} | {h['val_loss'][i]:8.4f} | {h['val_f1'][i]:8.4f} | {h['val_acc'][i]:8.4f} | {h['lr'][i]:.2e}")
