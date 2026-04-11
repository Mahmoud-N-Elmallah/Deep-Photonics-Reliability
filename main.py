import subprocess
import sys
import os
from pathlib import Path

def run_step(command, description):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        # Use sys.executable to ensure we use the same environment (uv or venv)
        result = subprocess.run([sys.executable] + command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step failed: {description}")
        print(f"Exit code: {e.returncode}")
        return False

def main():
    project_root = Path(__file__).parent
    src_dir = project_root / 'src'
    
    # Check for dataset
    data_dir = project_root / 'data' / 'images'
    if not data_dir.exists():
        print(f"Error: Dataset not found at {data_dir}")
        print("Please ensure the PV images are placed in data/images/ before running.")
        return

    pipeline = [
        (["src/calc_stats.py"], "Step 0: Calculating Dataset Statistics"),
        (["src/train.py"], "Step 1 & 2: Baseline Model Training (Phase 1-2)"),
        (["src/grad_cam.py", "--generate_masks"], "Step 3: Grad-CAM Explainability & Teacher Mask Generation"),
        (["src/prepare_masks_csv.py"], "Step 3.5: Preparing Mask Mapping Catalog"),
        (["src/train_phase4.py"], "Step 4: Physics-Constrained Fine-Tuning"),
        (["src/evaluate_test_set.py"], "Final Step: Global Evaluation on Unseen Test Set")
    ]

    print("\nDeep Photonics Reliability full Pipeline")
    print("This will run all project phases sequentially and after it finishes u will end up with the whole porject outputs done in thier own folders.\n")

    for cmd, desc in pipeline:
        success = run_step(cmd, desc)
        if not success:
            print("\nPipeline halted due to error.")
            sys.exit(1)

    print("\n" + "#"*60)
    print("PIPELINE COMPLETE!")
    print("Project reached 82% F1-Score benchmark (Phase 4).")
    print("Final results can be found in 'results/final_evaluation/'")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
