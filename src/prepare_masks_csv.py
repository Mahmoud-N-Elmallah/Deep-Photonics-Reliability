import pandas as pd
from pathlib import Path
import os

def prepare_masks_mapping():
    project_root = Path.cwd()
    mask_root = project_root / 'data' / 'pseudo_masks' / 'masks'
    
    mapping_results = []
    
    for split in ['train', 'val']:
        split_dir = mask_root / split
        if not split_dir.exists():
            print(f"Split directory {split} not found. Skipping...")
            continue
            
        print(f"Processing {split} masks...")
        masks = list(split_dir.glob('*.png'))
        
        for mask_path in masks:
            # Mask name format: cellXXXX_mask.png
            # name: cellXXXX.png
            mask_name = mask_path.name
            original_image_name = mask_name.replace('_mask.png', '.png')
            
            
            mapping_results.append({
                'image_name': original_image_name,
                'mask_rel_path': f"pseudo_masks/masks/{split}/{mask_name}",
                'split': split
            })
            
    df = pd.DataFrame(mapping_results)
    out_path = project_root / 'data' / 'pseudo_masks_mapping.csv'
    df.to_csv(out_path, index=False)
    print(f"Mapping complete! Saved {len(df)} records to {out_path}")

if __name__ == "__main__":
    prepare_masks_mapping()
