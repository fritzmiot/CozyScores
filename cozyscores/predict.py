# predict.py - Script to predict scores for unscored images using an ensemble of models
import os
import sys
from pathlib import Path

# --- 1. THE BRAINS: Tell Python where the root is FIRST ---
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- 2. THE LIBRARIES: Standard imports ---
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# --- 3. THE ENGINE: cozyscores imports now that the path is set ---
from cozyscores.utils import create_model, load_image, get_transforms, load_model, load_unscored_images, load_config

def predict_images(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths are already resolved as absolute by utils.load_config
    model_dir = cfg["model_dir"]
    unscored_dir = cfg["unscored_dir"]
    output_dir = cfg["output_dir"]
    ensemble_size = cfg["ensemble_size"]
    image_size = cfg["image_size"]

    # Load Ensemble
    models = []
    for i in range(ensemble_size):
        path = model_dir / f"model_{i}_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing ensemble model: {path}")
        model = create_model()
        model = load_model(model, None, path, device)[0]
        model.eval()
        models.append(model)
    
    image_paths = load_unscored_images(unscored_dir)
    if not image_paths:
        print(f"No images found in {unscored_dir}")
        return

    all_results = []
    transform = get_transforms(image_size=image_size, mode="val")

    for img_path in tqdm(image_paths, desc="Predicting"):
        img = load_image(img_path)
        if img is None: continue
        
        input_tensor = transform(img).to(device).unsqueeze(0)
        
        preds = []
        with torch.no_grad():
            for model in models:
                preds.append(model(input_tensor).item())
        
        relative_path = os.path.relpath(img_path, unscored_dir)
        all_results.append({
            "path_and_file": relative_path,
            "mean_score": np.mean(preds),
            "std_dev": np.std(preds),
            **{f"model_{i}": p for i, p in enumerate(preds)}
        })

    df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir / "predicted_scores.csv", index=False)
    print(f"Results saved to: {output_dir / 'predicted_scores.csv'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to project config.yaml")
    args = parser.parse_args()

    # Load merged config
    cfg = load_config(args.config)
    predict_images(cfg)

if __name__ == "__main__":
    main()