# evaluate.py - Script to evaluate ensemble model performance on validation (test) set
import os
import sys
from pathlib import Path

# --- 1. THE BRAINS: Tell Python where the root is first ---
root_dir = Path(__file__).resolve().parent.parent
if root_dir not in sys.path:
    sys.path.append(str(root_dir))

# --- 2. THE LIBRARIES: Standard imports ---
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# --- 3. THE ENGINE: cozyscores imports now that the path is set ---
from cozyscores.utils import create_model, load_image, get_transforms, load_model, load_unscored_images, load_config, NestDataset, load_labels

def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    df = load_labels(cfg["scored_dir"])
    _, val_df = train_test_split(df, test_size=1.0 - cfg["train_val_split"], random_state=42)

    val_loader = DataLoader(
        NestDataset(val_df, get_transforms(cfg["image_size"], mode="val")), 
        batch_size=cfg["batch_size"], 
        shuffle=False
    )

    # Load Ensemble
    models = []
    for i in range(cfg["ensemble_size"]):
        path = cfg["model_dir"] / f"model_{i}_best.pt"
        model = load_model(create_model(), None, path, device)[0]
        model.eval()
        models.append(model)
    
    all_preds, all_labels = [[] for _ in range(cfg["ensemble_size"])], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            for i, model in enumerate(models):
                all_preds[i].extend(model(images).squeeze().cpu().numpy())
            all_labels.extend(labels.numpy())

    preds = np.stack(all_preds)
    mean_preds, std_preds = preds.mean(axis=0), preds.std(axis=0)

    # Metrics
    print(f"\nResults: MSE: {mean_squared_error(all_labels, mean_preds):.4f} | R2: {r2_score(all_labels, mean_preds):.4f}")

    # Save
    res_df = pd.DataFrame({"actual": all_labels, "pred_mean": mean_preds, "pred_std": std_preds})
    res_df.to_csv(cfg["output_dir"] / "val_results.csv", index=False)
    
    # Plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=all_labels, y=mean_preds)
    plt.plot([1,5], [1,5], 'r--')
    plt.savefig(cfg["output_dir"] / f"val_plot_{datetime.now().strftime('%Y.%m.%d_%H%M')}.png")
    print(f"Saved results to {cfg['output_dir']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    evaluate(cfg)

if __name__ == "__main__":
    main()