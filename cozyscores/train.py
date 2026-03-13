# train.py - Script to train ensemble models for CozyScores
import os
import sys
from pathlib import Path

# --- 1. THE BRAINS: Tell Python where the root is FIRST ---
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- 2. THE LIBRARIES: Standard imports ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse

# --- 3. THE ENGINE: cozyscores imports now that the path is set ---
from cozyscores.utils import load_config, NestDataset, create_model, save_model, load_model, get_transforms, load_labels


def tukey_biweight_loss(pred, target, c=4.685):
    error = pred - target
    error_sq = (error / c) ** 2
    mask = error_sq < 1
    loss = torch.zeros_like(error_sq)
    loss[mask] = c**2 / 6 * (1 - (1 - error_sq[mask]) ** 3)
    loss[~mask] = c**2 / 6
    return loss.mean()

def train_model(cfg, model_idx, train_loader, val_loader, device):
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    best_loss = float("inf")
    patience_counter = 0
    
    # This ensures if ensemble_size is changed in config, the model files are still saved correctly
    os.makedirs(cfg["model_dir"], exist_ok=True)
    best_path = cfg["model_dir"] / f"model_{model_idx}_best.pt"

    for epoch in range(cfg["epochs"]):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = tukey_biweight_loss(model(imgs).squeeze(), lbls)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                val_loss += tukey_biweight_loss(model(imgs.to(device)).squeeze(), lbls.to(device)).item()
        val_loss /= len(val_loader)

        print(f"Model {model_idx} | Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, path=best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_labels(cfg["scored_dir"])
    train_df, val_df = train_test_split(df, test_size=1.0-cfg["train_val_split"], random_state=42)

    train_loader = DataLoader(NestDataset(train_df, get_transforms(cfg["image_size"], mode="train")), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(NestDataset(val_df, get_transforms(cfg["image_size"], mode="val")), batch_size=cfg["batch_size"])

    os.makedirs(cfg["output_dir"], exist_ok=True)

    for i in range(cfg["ensemble_size"]):
        print(f"\n--- Training Ensemble Member {i} ---")
        train_model(cfg, i, train_loader, val_loader, device)

if __name__ == "__main__":
    main()