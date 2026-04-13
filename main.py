import subprocess
import sys
import argparse
from pathlib import Path

def run_step(command, description, cwd):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        # Use sys.executable to ensure we use the same environment
        subprocess.run([sys.executable] + command, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step failed: {description}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Deep Photonics Reliability Pipeline Orchestrator")
    parser.add_argument("--phase", type=str, default="all", 
                        choices=["all", "1-2", "3", "4", "eval"],
                        help="Phase to run: 1-2, 3, 4, eval, or all (default)")
    parser.add_argument("--config", type=str, default="src/config.yaml", 
                        help="Path to the configuration file (default: src/config.yaml)")
    
    args = parser.parse_args()
    project_root = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
    
    # Check for dataset
    data_dir = project_root / 'data' / 'images'
    if not data_dir.exists():
        print(f"Error: Dataset not found at {data_dir}")
        print("Please ensure the PV images are placed in data/images/ before running.")
        return

    # Define all possible steps
    all_steps = {
        "0": (["src/calc_stats.py"], "Step 0: Calculating Dataset Statistics", True),
        "1-2": (["src/train.py"], "Step 1 & 2: Baseline Model Training (Phase 1-2)", True),
        "3": (["src/grad_cam.py", "--generate_masks"], "Step 3: Grad-CAM Explainability & Teacher Mask Generation", True),
        "3.5": (["src/prepare_masks_csv.py"], "Step 3.5: Preparing Mask Mapping Catalog", False),
        "4": (["src/train_phase4.py"], "Step 4: Physics-Constrained Fine-Tuning", True),
        "eval": (["src/evaluate_test_set.py"], "Final Step: Global Evaluation on Unseen Test Set", True)
    }

    # Determine pipeline based on phase
    if args.phase == "all":
        pipeline_keys = ["0", "1-2", "3", "3.5", "4", "eval"]
    elif args.phase == "1-2":
        pipeline_keys = ["0", "1-2"]
    elif args.phase == "3":
        pipeline_keys = ["3", "3.5"]
    elif args.phase == "4":
        pipeline_keys = ["4"]
    elif args.phase == "eval":
        pipeline_keys = ["eval"]
    else:
        pipeline_keys = []

    pipeline = [all_steps[k] for k in pipeline_keys]

    print(f"\nDeep Photonics Reliability Pipeline Orchestrator")
    print(f"Mode: {'Full Run' if args.phase == 'all' else f'Phase {args.phase}'}")
    print(f"Config: {config_path}\n")

    for cmd, desc, accepts_config in pipeline:
        command = list(cmd)
        if accepts_config:
            command.extend(["--config", str(config_path)])
        success = run_step(command, desc, project_root)
        if not success:
            print("\nPipeline halted due to error.")
            sys.exit(1)

    print("\n" + "#"*60)
    print("PIPELINE EXECUTION COMPLETE!")
    if args.phase in ["all", "eval"]:
        print("Final results can be found in 'results/final_evaluation/'")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
