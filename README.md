# 🐭 CozyScores: Deep Learning Nest Quality Scoring

![CozyScores Pipeline](CozyScores_Pipeline.png)

CozyScores is a PyTorch-based pipeline for automated, high-throughput scoring of mouse nesting behavior. It uses an ensemble of ResNet18-based regression models to predict a continuous nest quality score from 1.0–5.0 and reports ensemble standard deviation as an uncertainty/confidence metric.

CozyScores is being developed as both a scoring tool and a broader analysis pipeline for nesting behavior experiments, including model interpretability, Grad-CAM visualization, versioned model workflows, and future nest-morphology metrics.

---

## 🛠️ Environment Setup

1. **Create the Environment:**
    To ensure CozyScores runs with the expected library versions, set up a local conda environment.
```bash
conda create -n cozyscores_env python=3.10 -y
conda activate cozyscores_env
```

    
2. **Install Dependencies:**
    Navigate to your local CozyScores directory and run:
    ```Bash
    # Replace the path below with your actual folder location
    cd path/to/your/CozyScores
    pip install -r requirements.txt
    ```   

3. **Git LFS:**
    If using Git LFS-tracked model weights, also install and pull LFS files:
```bash
git lfs install
git lfs pull
```
---
## 📂 Project Structure

CozyScores uses a central-engine architecture. Core scripts live in `cozyscores/`, trained model weights live in `models/`, and experiment-specific inputs/outputs live in `projects/`.
```text
CozyScores/
├── cozyscores/                         # Central engine
│   ├── init_project.py                 # Initialize a new project folder
│   ├── utils.py                        # Config loading, datasets, transforms, model helpers
│   ├── train.py                        # Train ensemble scoring models
│   ├── predict.py                      # Predict scores for unscored photos
│   ├── evaluate.py                     # Evaluate validation-set performance
│   ├── gradcam.py                      # Generate Grad-CAM interpretability maps
│   ├── nest_emphasis_qc.py             # QC nest-emphasis preprocessing
│   └── base_config.yaml                # Default parameters and path logic
│
├── models/
│   └── scoring/
│       ├── v3/                         # Archived model version
│       ├── v4/                         # Production/reference model
│       └── v5/                         # Nest-emphasis + random-erasing model
│
└── projects/
    └── SKA31/
        ├── config.yaml                 # Project-specific config overrides
        ├── data/
        │   ├── scored_photos/          # Human-scored training/model-development photos
        │   ├── unscored_photos/        # Inference/study photos
        │   ├── score_keys/             # Approved example/reference images
        │   ├── metadata/               # Future experiment/image/animal metadata inputs
        │   └── segmentation_annotations/
        │       ├── images/             # Future segmentation annotation images
        │       ├── masks/              # Future binary nest-material masks
        │       └── annotation_manifest.csv
        ├── work/                       # Generated intermediate files/cache
        └── results/
            ├── predictions/            # Versioned prediction CSVs
            ├── gradcam/                # Versioned Grad-CAM outputs
            ├── evaluation/             # Versioned validation outputs
            ├── segmentation_qc/        # Future mask/segmentation QC
            ├── metrics/                # Future nest morphometrics
            ├── analysis/               # Future analysis-ready merged outputs
            └── figures/                # Future figure exports
```

Private training images, study photos, generated outputs, and intermediate files are ignored by Git by default. Only explicitly approved demo/reference images should be tracked.
---
## ⚙️ Configuration Magic

CozyScores uses a **Two-Tier Configuration** system:

1. **`cozyscores/base_config.yaml`:** The base config stores default training parameters, preprocessing settings, and path roots.
    
2. **Project `projects/YOUR_PROJECT/config.yaml`**: Located in your project folder. It only needs to contain things specific to that project i.e., values that differ from the base config. It automatically inherits everything else from the base.

For example:
```yaml
project_name: "SKA31"
scoring_model_version: "v5"
```

The scoring_model_version field automatically controls synchronized paths. For example:
```yaml
scoring_model_version: "v5"
```

automatically resolves to:
```yaml
models/scoring/v5/
projects/SKA31/results/predictions/scoring_v5/
projects/SKA31/results/gradcam/scoring_v5/
projects/SKA31/results/evaluation/scoring_v5/
```

To switch between model versions, change only:
```yaml
scoring_model_version: "" # Type desired model name inside ""
```
---
## 🚀 Usage Guide

Run all commands from the root CozyScores/ directory.

### 1. Preparing Data for Training

To train the model, your `scored_photos/` directory **must** use subfolders, each with a `scores.csv` and corresponding images.

Example:
```text
scored_photos/
├── Photos 01/
│   ├── scores.csv
│   ├── IMG_001.HEIC
│   └── ...
├── BONUS/
│   ├── scores.csv
│   └── ...
```

- **CSV Requirement:** Must have a column named `filename` (or `image`) and `average_score` (or `score`).
    
- **Format Support:** Native support for `.jpg`, `.png`, and `.heic` (iPhone photos supported with pillow-heif).

#### Ignoring Specific Data Folders

To temporarily exclude a source folder without moving it, place an empty marker file inside that folder:
```text
.cozyscores_ignore
```

For example:
```text
scored_photos/BONUS/.cozyscores_ignore
```

CozyScores will skip ignored folders during loading (convenient for quick test with subsets instead of full image library).

### 2. Training an Ensemble

Train multiple models at once to ensure statistical robustness.

```Bash
python cozyscores/train.py --config projects/YOUR_PROJECT/config.yaml
```

- Uses **Tukey Biweight Loss** to remain robust against outlier manual scores.
    
- Automatically performs an 80/20 train/validation split.
    
- Saves the best weights for each ensemble member to `models/v4/`.

Training uses:

- ResNet18 backbone pretrained on ImageNet

- scaled sigmoid output constrained to 1.0–5.0

- Tukey biweight loss

- 80/20 image-level train/validation split

- ensemble training based on ensemble_size

- optional nest-emphasis augmentation

- optional random erasing augmentation

Best model weights are saved to the versioned model directory:
```text
models/scoring/<scoring_model_version>/
```

Example:
```text
models/scoring/v5/model_0_best.pt
models/scoring/v5/model_1_best.pt
...
```

### 3. Evaluation & Validation

Evaluate validation-set performance (internal), or how well the AI matches human intuition/consensus.

```Bash
python cozyscores/evaluate.py --config projects/YOUR_PROJECT/config.yaml
```

Outputs are saved to:
```text
projects/SKA31/results/evaluation/scoring_<version>/
```

Generates:

- actual-vs-predicted plot.
    
- Calculates $R^2$

- Mean Squared Error (MSE).

- WIP: more to come...  
    

### 4. Predicting New Scores (Inference)

The primary tool for your research. Predict scores for images in `unscored_photos/`:

```Bash
python cozyscores/predict.py --config projects/YOUR_PROJECT/config.yaml
```

Outputs are saved to:
```text
projects/SKA31/results/predictions/scoring_<version>/predicted_scores.csv
```
    
Confidence Metrics:

- relative image path

- individual model predictions and ensemble `mean_score` 

- ensemble `std_dev` 

A high ensemble standard deviation indicates greater model disagreement (models are "unsure") and may warrant manual review.
    
### 5. Grad-CAM Explainability Maps

Generate Grad-CAM maps to visualize image regions contributing to the model output:

```Bash
python cozyscores/gradcam.py --config projects/YOUR_PROJECT/config.yaml
```

Availale visualization modes:

```bash
python cozyscores/gradcam.py --config projects/SKA31/config.yaml --mode single
python cozyscores/gradcam.py --config projects/SKA31/config.yaml --mode average
python cozyscores/gradcam.py --config projects/SKA31/config.yaml --mode panel
```

Modes:

- single: uses the first ensemble member
- average: averages Grad-CAM heatmaps across ensemble members
- panel: produces side-by-side Grad-CAM panels for all ensemble members

Outputs are saved to:
```text
projects/SKA31/results/gradcam/scoring_<version>/
```

These visualizations help verify the CNN is relying primarily on the biologically relevant feature rather than image artifacts or shortcut features.

### 6. Nest-Emphasis QC

CozyScores includes a nest-emphasis preprocessing workflow intended to reduce shortcut learning by visually de-emphasizing bedding, cage walls, sticky notes, and other contextual artifacts while preserving likely white nesting material.

Run QC examples:

```bash
python cozyscores/nest_emphasis_qc.py --config projects/SKA31/config.yaml --n 30
```

This is primarily a development/QC tool, not a final analysis output.
---
## 🔬 Core Technologies

- **Framework:** PyTorch

- **Backbone:** ResNet18 pre-trained on ImageNet

- **Prediction:** Continuous regression, not classification
    
- **Output scaling:** Scaled sigmoid constrains predictions to 1.0–5.0
    
- **Preprocessing:** Letterbox resizing to preserve cage aspect ratio

- **Loss:** Tukey biweight loss

- **Uncertainty estimation:** Ensemble standard deviation

- **Interpretability:** Custom Grad-CAM implementation for CNN saliency visualization and shortcut-learning inspection

- **Illustrations:** Figure created with BioRender.com
    
---
## 🧭 Development Roadmap

Planned/potential future features include:

- supervised nest-material segmentation
- mask-aware scoring models
- nest area, dispersion, compactness, and connected-component metrics
- nest centroid / center-of-mass tracking across time
- cage-normalized coordinate mapping
- perspective correction / top-down geometric normalization
- colony-wide nest location heatmaps
- metadata-aware analysis pipelines
- automated time-series trajectory plotting
- analysis-ready merged output tables
- publication-ready figure generation
- GUI/Desktop application

The long-term goal is for CozyScores to first localize nesting material, then derive both a continuous score and interpretable nest-organization metrics from the segmented nest structure.

---
## 📦 Git / Data Management Notes

By default, .gitignore excludes:

- private training images
- study images
- generated results
- intermediate working files
- metadata tables
- raw segmentation annotations

Approved demo images can be explicitly unignored.

Model weights are large and should be handled with Git LFS:
```bash
git lfs install
git lfs track "models/scoring/**/*.pt"
```
---
## 🎓 Citation

If you use CozyScores in your research, please cite:
> Miot & Shih et al (TBD). TITLE. [In Review/Journal].

---
## ⚖️ License & Acknowledgements
- Distributed under the **MIT License**. See `LICENSE` for more information.
- Illustrations created with **BioRender.com**.
- Lab volunteers who helped with nesting-material preparation and human scoring.

---
## 📝 Lab Notes & Troubleshooting

To start a new project, simply copy the `projects/SKA-31` folder structure and update the paths in the new `config.yaml`.

Python cannot find cozyscores?

Run scripts from the root directory:

```bash
cd path/to/CozyScores
python cozyscores/predict.py --config projects/SKA31/config.yaml
```

HEIC files will not open?

Confirm dependencies are installed:

```bash
pip install pillow-heif
```

Results are being overwritten or mixed between model versions?

Check the project config:

```yaml
scoring_model_version: "v5"
```

Outputs are automatically routed to versioned result folders such as:

```text
results/predictions/scoring_v5/
results/gradcam/scoring_v5/
results/evaluation/scoring_v5/
```

A folder should be skipped during loading?

Place an empty file named:

```
.cozyscores_ignore
```

inside the folder to exclude it from image loading.