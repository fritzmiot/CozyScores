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

    output_dir = cfg.get("evaluation_output_dir", cfg["output_dir"])
    scoring_model_version = cfg.get("scoring_model_version", "unknown")

    print(f"Scoring model version: {scoring_model_version}")
    print(f"Loading models from: {cfg['model_dir']}")
    print(f"Saving evaluation outputs to: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    df = load_labels(cfg["scored_dir"])

    _, val_df = train_test_split(
        df,
        test_size=1.0 - cfg["train_val_split"],
        random_state=42
    )

    val_loader = DataLoader(
        NestDataset(val_df, get_transforms(cfg["image_size"], mode="val")),
        batch_size=cfg["batch_size"],
        shuffle=False
    )

    # Load ensemble models.
    models = []
    for i in range(cfg["ensemble_size"]):
        path = cfg["model_dir"] / f"model_{i}_best.pt"

        if not path.exists():
            raise FileNotFoundError(f"Missing ensemble model: {path}")

        model = load_model(create_model(), None, path, device)[0]
        model.eval()
        models.append(model)

    all_preds = [[] for _ in range(cfg["ensemble_size"])]
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            for i, model in enumerate(models):
                outputs = model(images).squeeze()
                all_preds[i].extend(np.atleast_1d(outputs.cpu().numpy()))

            all_labels.extend(labels.numpy())

    preds = np.stack(all_preds)
    mean_preds = preds.mean(axis=0)
    std_preds = preds.std(axis=0)

    mse = mean_squared_error(all_labels, mean_preds)
    mae = mean_absolute_error(all_labels, mean_preds)
    r2 = r2_score(all_labels, mean_preds)

    print(
        f"\nResults: "
        f"MSE: {mse:.4f} | "
        f"MAE: {mae:.4f} | "
        f"R2: {r2:.4f}"
    )

    # Save row-level validation results.
    res_df = pd.DataFrame({
        "actual": all_labels,
        "pred_mean": mean_preds,
        "pred_std": std_preds,
        "scoring_model_version": scoring_model_version,
    })

    for i in range(cfg["ensemble_size"]):
        res_df[f"model_{i}"] = preds[i]

    results_path = output_dir / "val_results.csv"
    res_df.to_csv(results_path, index=False)

    # Save summary metrics.
    summary_df = pd.DataFrame([{
        "scoring_model_version": scoring_model_version,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "n_validation_images": len(all_labels),
    }])

    summary_path = output_dir / "val_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Plot actual vs predicted.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=all_labels, y=mean_preds)
    plt.plot([1, 5], [1, 5], 'r--')
    plt.xlabel("Human consensus score")
    plt.ylabel("CozyScores predicted score")
    plt.title(f"Validation: {scoring_model_version}")
    plt.tight_layout()

    plot_path = output_dir / f"val_plot_{datetime.now().strftime('%Y.%m.%d_%H%M')}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved validation results to: {results_path}")
    print(f"Saved validation summary to: {summary_path}")
    print(f"Saved validation plot to: {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    evaluate(cfg)

if __name__ == "__main__":
    main()
# EoF evaluate.py