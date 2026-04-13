import torch
import pandas as pd
import pickle
from pathlib import Path

def load_history_artifact(path):
    artifact_path = Path(path)
    if artifact_path.name == 'training_history.pkl':
        with open(artifact_path, 'rb') as f:
            return pickle.load(f)
    payload = torch.load(artifact_path, map_location='cpu', weights_only=False)
    return payload.get('history', payload)

def detailed_analysis(checkpoint_path):
    history = load_history_artifact(checkpoint_path)
    df = pd.DataFrame(history)
    
    print("Full History:")
    print(df.to_string())

if __name__ == "__main__":
    detailed_analysis(r"checkpoints/tri_channel/training_history.pkl")
