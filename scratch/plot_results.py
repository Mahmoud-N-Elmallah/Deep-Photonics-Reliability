import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training(checkpoint_path, output_img):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    history = checkpoint.get('history', {})
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Plot
    axes[0].plot(df['train_loss'], label='Train Loss')
    axes[0].plot(df['val_loss'], label='Val Loss')
    axes[0].set_title('Loss History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # F1 Plot
    axes[1].plot(df['train_f1'], label='Train F1')
    axes[1].plot(df['val_f1'], label='Val F1')
    axes[1].set_title('F1 Score History')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    plot_training(r"c:\Projects\PV Cells Fault CV\Deep-Photonics-Reliability\checkpoints\tri_channel\latest_checkpoint.pkl", 
                  r"c:\Projects\PV Cells Fault CV\Deep-Photonics-Reliability\scratch\training_plot.png")
