# Deep Photonics Reliability: Physics-Constrained PV Fault Analysis

## Project Mission: Bridging Physics and Deep Learning
Deep Photonics Reliability is a comprehensive research pipeline designed to solve the **"Black Box" Problem** in Photovoltaic (PV) Electroluminescence (EL) image classification. Standard deep learning models often optimize based on visual artifacts unrelated to the actual physics of the cell (e.g., background noise). 

This project implements a **Physics-Constrained Supervision** framework. It forces the model to ignore deterministic manufacturing patterns and strictly prioritize the spatial geometry of structural anomalies, such as micro-cracks and material inclusions. The pipeline follows a curriculum-based learning approach across four distinct phases.

---

## 1. The Physics of Electroluminescence (EL)
Electroluminescence is the phenomenon where a material emits light in response to the passage of an electric current. In silicon solar cells, this occurs via **Radiative Recombination**.
*   **The Baseline**: A healthy PV cell emits a uniform infrared glow (captured by InGaAs cameras).
*   **The Anomaly**: Structural defects like micro-cracks or Potential Induced Degradation (PID) act as "sinks" for charge carriers. This leads to **Non-Radiative Recombination**, resulting in dark pixels or "shunts" in the EL image.
*   **The Challenge**: EL images are "busy." They contain a dominant, deterministic grid of busbars and fingers that can mislead standard neural networks into false correlations.

---

## 2. Research Roadmap: A Textbook Curriculum

### Phase 1: Spectral Data Engineering (FFT)
**Theory**: PV cells are periodic structures. In the frequency domain (Fourier Space), the regular grid of metal fingers appears as high-magnitude spikes at specific coordinates.
*   **Process**: We apply a **2D Fast Fourier Transform (FFT)** to shift the signal into the frequency domain.
*   **Notch Filtering**: A Gaussian notch filter is applied to suppress the periodic frequencies (the grid).
*   **Reconstruction**: After an Inverse FFT (IFFT), we obtain a "cleaned" image where the deterministic grid is suppressed, but the **stochastic defects** remain prominent.
*   **Significance**: This reduces the "signal-to-noise" ratio for the model, forcing it to look for deviations from periodicity.

### Phase 2: Tri-Channel Feature Fusion
**Architecture**: We don't just feed the raw image. We create a **Synthetic Triple-Channel Input**:
1.  **Channel 1 (Raw)**: Preserves the context of the whole cell.
2.  **Channel 2 (FFT-Cleaned)**: Highlights the stochastic anomalies.
3.  **Channel 3 (Enhanced)**: Utilizes Contrast Limited Adaptive Histogram Equalization (CLAHE) to sharpen the edges of cracks.
*   **Result**: The model (PhotonicResNet18) learns to correlate signatures across different spectral and contrast domains simultaneously.

### Phase 3: Explainability as a Diagnostic (Grad-CAM)
**Theory**: Before we trust the model, we must audit it. We utilize **Gradient-weighted Class Activation Mapping (Grad-CAM)**.
*   **Mechanism**: We calculate the gradients of the target class score with respect to the feature maps of the final convolutional layer.
*   **Teacher Generation**: By thresholding these activations, we generate **Pseudo-Masks**. These masks represent "where the model is looking." 
*   **Data Audit**: We identified that purely statistical models often "hallucinate" (look at background corners). This discovery led to Phase 4.

### Phase 4: Physics-Constrained Optimization
**Theory**: Instead of letting the model look wherever it wants, we enforce a **Spatial Loss Constraint**.
*   **The Loss Function**: We combine Standard Cross-Entropy with a **Dice-BCE Hybrid Loss**.
*   **Supervision**: We use the high-quality pseudo-masks from Phase 3 as "Spatial Targets." If the model's internal attention deviates from the physical path of the defect, the Dice Loss penalizes it.
*   **Confidence Gate**: We implement **Confidence-Weighted Supervision**. If the model is unsure (Low Softmax probability), the physics constraint is relaxed to prevent learning from noise.

---

## 3. Analysis & Visual Evidence

### A. Raw Dataset Primer: The Severity Scale
| Normal (0.00) | Minor-Defect (0.33) | Moderate-Defect (0.67) | Major-Defect (1.00) |
*   **Elaboration**: The dataset categorization is based on a probability scale. While it focuses on structural cracks, the labels reflect the likelihood of the defect impacting the overall module efficiency.

### B. Phase 3: Automated Teacher-Mask Generation
| | | | |
| :---: | :---: | :---: | :---: |
| ![CAM 0013](data/pseudo_masks/visuals/train/cell0013_cam.jpg) | ![CAM 0031](data/pseudo_masks/visuals/train/cell0031_cam.jpg) | ![CAM 0112](data/pseudo_masks/visuals/train/cell0112_cam.jpg) | ![CAM 0115](data/pseudo_masks/visuals/train/cell0115_cam.jpg) |
| ![CAM 0171](data/pseudo_masks/visuals/train/cell0171_cam.jpg) | ![CAM 0242](data/pseudo_masks/visuals/train/cell0242_cam.jpg) | ![CAM 0375](data/pseudo_masks/visuals/train/cell0375_cam.jpg) | ![CAM 0438](data/pseudo_masks/visuals/train/cell0438_cam.jpg) |

*   **Elaboration**: These samples illustrate the automated extraction of anomaly paths using Grad-CAM. By setting thresholds on activations, the system generates localized "teacher masks" that provide the ground-truth guidance for Phase 4.

### C. The Quality Filter (Solving Noise)
![Bad Mask Sample](data/pseudo_masks/visuals/train/cell0508_cam.jpg)
*   **Analysis**: Sample 0508 represents a "hallucinated" mask. Note how the focus is diffuse across the whole cell rather than on a structural line. Our **Phase 4 Quality Filter** automatically identifies and ignores such masks to prevent the propagation of visual noise.

### D. Phases Comparison: "Attention Sharpening"
| Comparison Sample 0010 | Comparison Sample 0517 |
| :---: | :---: |
| ![Phase Comparison 0010](data/phase_comparison/cell0010_comparison.jpg) | ![Phase Comparison 0517](data/phase_comparison/cell0517_comparison.jpg) |

*   **Interpretation**: In Phase 3 (Standard), activations are broad and "leaky." In Phase 4 (Physics-Aware), the attention maps are **sharpened** and strictly locked onto the physical structural paths. This is the result of **Quadratic Attention Sharpening** integrated into the forward pass.

---

## 4. Final Evaluation (Generalization)

### Comprehensive Blind-Test Gallery
| Case 0 | Case 1 | Case 2 | Case 3 |
| :---: | :---: | :---: | :---: |
| ![T0](results/final_evaluation/blind_test_0.jpg) | ![T1](results/final_evaluation/blind_test_1.jpg) | ![T2](results/final_evaluation/blind_test_2.jpg) | ![T3](results/final_evaluation/blind_test_3.jpg) |
| **Case 4** | **Case 5** | **Case 6** | **Case 7** |
| ![T4](results/final_evaluation/blind_test_4.jpg) | ![T5](results/final_evaluation/blind_test_5.jpg) | ![T6](results/final_evaluation/blind_test_6.jpg) | ![T7](results/final_evaluation/blind_test_7.jpg) |
| **Case 8** | **Case 9** | **Case 10** | **Case 11** |
| ![T8](results/final_evaluation/blind_test_8.jpg) | ![T9](results/final_evaluation/blind_test_9.jpg) | ![T10](results/final_evaluation/blind_test_10.jpg) | ![T11](results/final_evaluation/blind_test_11.jpg) |

*   **Statistical Breakdown**: The model achieved a **Weighted F1-Score typically ranging from 0.77 to 0.82**. The high precision on "Major Defects" indicates that the physics constraint successfully eliminated false positives triggered by background cell structures.

---

## 5. Mathematical Foundation
The total objective function optimized in Phase 4 is:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{physics} \cdot \left[ \text{Dice}(A, M) + \text{BCE}(A, M) \right]$$
Where:
*   $A$: The Quadratic Attention Map ($A = \text{softmax}( \text{conv}(x) )^2$).
*   $M$: The Physics-Guided Pseudo-Mask.
*   $\lambda_{physics}$: Ramps from 0.0 to 0.25 over a 5-epoch warmup to prevent catastrophic forgetting.

---

## 6. Training Dynamics
![Training Curves](scratch/training_plot.png)
*   **Elaboration**: This plot establishes the baseline convergence for tri-channel EL classification before the introduction of physical regularization.

---

## 7. Data Augmentation & Resilience
**Theory**: PV defect shapes are stochastic but follow certain geometric physical constraints (cracks are typically straight or jagged lines). 
*   **Elastic Transformations**: We utilize Elastic Distortions to simulate physical variations in material stress. This forces the model to learn the *topology* of a crack rather than memorizing specific coordinates.
*   **Joint Transform Strategy**: In Phase 4, we ensure both the **Input Image** and the **Teacher Mask** undergo identical geometric transformations in sync. This preserves the spatial fidelity of the physics constraint.

## 8. Performance Metrics: Handling Class Imbalance
**Theory**: In PV cell production, "Normal" cells significantly outnumber defective ones. This leads to a class imbalance problem where high Accuracy can be misleading.
*   **Weighted F1-Score**: We prioritize the F1-Score (the harmonic mean of Precision and Recall) as our primary benchmark. 
*   **Significance**: A high F1-Score ensures the model is both "Precise" (doesn't flag finger lines as cracks) and has high "Recall" (doesn't miss subtle micro-cracks that could lead to field failures).

---

## 9. Project Architecture
```text
├── main.py                     # Unified Pipeline Orchestrator (Single-entry point)
├── README.md                   # Textbook-style Documentation
├── .gitignore                  # Git Ignore rules (Selectively un-ignores visuals)
├── data/ (not tracked due to size)
│   ├── images/                 # Raw EL images (Input Directory)
│   ├── pseudo_masks/           # Generated during Phase 3
│   │   ├── masks/              # Binary numpy/image masks for Phase 4 training
│   │   └── visuals/            # Annotated Grad-CAM overlays for auditing
│   ├── phase_comparison/       # Side-by-side progression analysis (P3 vs P4)
│   ├── train_data.csv          # Catalog (Probability labels: 0.0, 0.33, 0.67, 1.0)
│   └── pseudo_masks_mapping.csv # Catalog mapping images to Phase 3 masks
├── src/
│   ├── calc_stats.py           # Dataset normalization calculator
│   ├── config.yaml             # Centralized hyperparameter configuration
│   ├── evaluate_test_set.py    # Final evaluation & reporting script
│   ├── grad_cam.py             # Feature visualization & mask extraction
│   ├── model.py                # PhotonicResNet18 Architecture (with Attention)
│   ├── physics_utils.py        # FFT & Signal processing utilities
│   ├── train_phase4.py         # Physics-constrained training entry
│   ├── training_engine.py      # Core trainer with Multi-objective Loss
│   └── utils.py                # Shared helpers (Loss, Optim, Schedulers)
├── checkpoints/ (not tracked due to size)
└── results/
    └── final_evaluation/       # Test reports, Confusion matrices, & Curves
```

---
**Mahmoud-N-Elmallah**
*Advancing reliability in renewable energy through Physics-Aware Machine Learning.*
