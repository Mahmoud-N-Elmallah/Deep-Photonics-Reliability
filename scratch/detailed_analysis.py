import torch
import pandas as pd

def detailed_analysis(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    history = checkpoint.get('history', {})
    df = pd.DataFrame(history)
    
    print("Full History:")
    print(df.to_string())

if __name__ == "__main__":
    detailed_analysis(r"checkpoints/tri_channel/latest_checkpoint.pkl")
