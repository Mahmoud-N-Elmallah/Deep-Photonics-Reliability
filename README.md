# Deep Photonics Reliability: Physics-Constrained PV Fault Analysis

## Overview
Deep Photonics Reliability is a state-of-the-art deep learning pipeline documentation for solving the "Black Box" challenge in Photovoltaic (PV) cell Electroluminescence (EL) image classification. Unlike standard classifiers that may optimize for non-physical background artifacts, this project implements **Physics-Constrained Supervision** to enforce spatial focus on the physical geometry of structural defects such as micro-cracks, Potential Induced Degradation (PID), and busbar anomalies.

---

## Roadmap

### Phase 1: Signal-Aided Data Engineering (FFT)
*   **Significance**: PV cells exhibit periodic manufacturing structures. Filtering this periodicity in the frequency domain allows the model to prioritize stochastic deviations (faults) over the deterministic background grid.

### Phase 3 & 4: From Explainability to Spatial Constraint
*   **Significance**: Phase 3 utilized Gradient-weighted Class Activation Mapping (Grad-CAM) to audit internal feature attention. Phase 4 operationalized these insights as a supervised constraint. By integrating a **Dice Loss** component into the objective function, the model transitioned from purely statistical classification to **Spatially-Aware Reliability**.

---

## Analysis & Visual Evidence

### 1. Training Dynamics (Phase 3 Baseline)
![Training Curves](scratch/training_plot.png)
*   **Elaboration**: This plot illustrates the baseline convergence of the PhotonicResNet18 architecture. It establishes the benchmark for tri-channel EL classification before the introduction of physical regularization.

### 2. Statistical Metrics (Confusion Matrix)
![Confusion Matrix](results/final_evaluation/confusion_matrix.png)
*   **Elaboration**: The diagonal dominance indicates robust classification across all fault categories. The model maintains high precision in **Normal** and **Busbar** identification, reflecting stable feature extraction in high-noise environments.

### 3. Phase 3: Automated Teacher-Mask Generation
![Grad-CAM Samples](data/pseudo_masks/visuals/train/cell0013_cam.jpg)
![Grad-CAM Samples](data/pseudo_masks/visuals/train/cell0031_cam.jpg)
![Grad-CAM Samples](data/pseudo_masks/visuals/train/cell0112_cam.jpg)
![Grad-CAM Samples](data/pseudo_masks/visuals/train/cell0115_cam.jpg)
![Grad-CAM Samples](data/pseudo_masks/visuals/train/cell0171_cam.jpg)
![Grad-CAM Samples](data/pseudo_masks/visuals/train/cell0242_cam.jpg)
![Grad-CAM Samples](data/pseudo_masks/visuals/train/cell0375_cam.jpg)
![Grad-CAM Samples](data/pseudo_masks/visuals/train/cell0438_cam.jpg)

*   **Elaboration**: These samples illustrate the automated extraction of defect locations using Gradient-weighted Class Activation Mapping. By setting a high percentile threshold on activations, the system generates localized masks that provide the ground-truth guidance for the subsequent physics constraint.

#### The "Bad Mask" Case (Noise Analysis)
![Bad Mask Sample](data/pseudo_masks/visuals/train/cell0508_cam.jpg)
*   **Elaboration**: Sample 0508 represents a "hallucinated" mask where the model's focus is diffuse or triggered by background textures. This specific case provided the scientific justification for our **Phase 4 Quality Filter**, which automatically identifies and ignores masks with excessive coverage or low confidence to prevent the propagation of visual noise.

### 4. Phases Comparison: "The Attention Sharpening"
![Phase Comparison Cell 0010](data/phase_comparison/cell0010_comparison.jpg)
![Phase Comparison Cell 0517](data/phase_comparison/cell0517_comparison.jpg)
*   **Elaboration**: Comparison of attention focus between Phase 3 (Base) and Phase 4 (Physics-Aware):
    *   **Phase 3 (Standard)**: Demonstrates diffuse activation, often broad or influenced by global cell features.
    *   **Phase 4 (Physics-Aware)**: Characterized by precise localization centered on the physical defect path. This is achieved through Quadratic Attention Scaling and Multi-Objective Loss Optimization.

### 5. Blind-Test Inference (Unseen Data)

| Sample 0 | Sample 1 |
| :---: | :---: |
| ![Test Sample 0](results/final_evaluation/blind_test_0.jpg) | ![Test Sample 1](results/final_evaluation/blind_test_1.jpg) |
| **Sample 2** | **Sample 3** |
| ![Test Sample 2](results/final_evaluation/blind_test_2.jpg) | ![Test Sample 3](results/final_evaluation/blind_test_3.jpg) |

*   **Elaboration**: These **Blind Tests** demonstrate generalization to totally novel data across various fault conditions. The consistent localization of faults (ranging from micro-cracks to PID clusters) confirms that the architecture has internalized robust physical features rather than specific training-set noise. The high-confidence activations on unseen samples indicate that the physics constraint successfully regularized the feature space.

---

## Project Architecture
```text
├── main.py                     # Unified Pipeline Orchestrator (Single-entry point)
├── README.md                   # Project Documentation
├── .gitignore                  # Git Ignore rules (Selectively un-ignores visuals)
├── data/ (the original dataset is not tracked due to size)
│   ├── images/                 # Raw EL images (Input Directory)
│   ├── pseudo_masks/           # Generated during Phase 3
│   │   ├── masks/              # Binary numpy/image masks for Phase 4 training
│   │   └── visuals/            # Annotated Grad-CAM overlays for auditing
│   │       ├── train/          # Training set visuals
│   │       └── val/            # Validation set visuals
│   ├── phase_comparison/       # Side-by-side progression analysis (P3 vs P4)
│   ├── train_data.csv          # Training split catalog
│   ├── val_data.csv            # Validation split catalog
│   ├── test_data.csv           # Final unseen test catalog
│   └── pseudo_masks_mapping.csv # Catalog mapping images to Phase 3 masks
├── src/
│   ├── calc_stats.py           # Dataset normalization calculator
│   ├── config.yaml             # Centralized hyperparameter configuration
│   ├── data_pipeline.py        # Tri-channel loading & augmentation engine
│   ├── dataset.py              # Custom PyTorch Datasets (Standard & Physics)
│   ├── evaluate_test_set.py    # Final evaluation & reporting script
│   ├── grad_cam.py             # Feature visualization & mask extraction
│   ├── model.py                # PhotonicResNet18 Architecture (with Attention)
│   ├── physics_utils.py        # FFT & Signal processing utilities
│   ├── prepare_masks_csv.py    # Catalog generation for Phase 4
│   ├── train.py                # Baseline training entry (Phase 1-2)
│   ├── train_phase4.py         # Physics-constrained training entry
│   ├── training_engine.py      # Core trainer with Multi-objective Loss
│   ├── utils.py                # Shared helpers (Loss, Optim, Schedulers)
│   ├── image_processing.ipynb  # Interactive signal processing lab
│   └── data_preparation.ipynb  # Interactive data preparation lab
├── checkpoints/ (not tracked due to size)
│   ├── tri_channel/            # Best baseline model weights
│   └── phase4_physics/         # Physics-constrained model weights & history
├── results/
│   └── final_evaluation/       # Test reports, Confusion matrices, & Curves
└── scratch/                    # Temporary logs and baseline plots
```

---

## Mathematical Foundation
The optimization process is formalized as a multi-objective task:
$$\mathcal{L}_{total} = \mathcal{L}_{Class} + \lambda \cdot [Dice(Attn, Mask) + BCE(Attn, Mask)]$$
By dynamically weighting the spatial regularization term ($\lambda$) based on Classification Confidence, the system ensures that spatial constraints are only enforced for high-probability class associations, thereby preventing the regularization of low-confidence feature noise.

---
**Mahmoud-N-Elmallah**
*Advancing reliability in renewable energy through Physics-Aware Machine Learning.*
