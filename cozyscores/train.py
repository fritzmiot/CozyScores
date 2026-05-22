# train.py - Script to train ensemble models for CozyScores
import os
import sys
from pathlib import Path

# --- 1. THE BRAINS: Tell Python where the root is FIRST ---
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- 2. THE LIBRARIES: Standard imports ---
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# --- 3. THE ENGINE: cozyscores imports now that the path is set ---
from cozyscores.utils import (
    load_config,
    NestDataset,
    create_model,
    save_model,
    get_transforms,
    load_labels,
    load_image,
    create_nest_emphasized_image
)


def tukey_biweight_loss(pred, target, c=4.685):
    """
    Tukey's biweight loss.

    Current v5 priority is interpretability/Grad-CAM behavior, so we are
    keeping the same loss as the prior version for now and changing only
    the training data emphasis/augmentation.
    """
    error = pred - target
    error_sq = (error / c) ** 2

    mask = error_sq < 1

    loss = torch.zeros_like(error_sq)

    loss[mask] = c**2 / 6 * (1 - (1 - error_sq[mask]) ** 3)

    loss[~mask] = c**2 / 6

    return loss.mean()


def get_nested_config(cfg, section_name, defaults):
    """
    Small helper for reading nested YAML config sections safely.

    Example:
        cfg["nest_emphasis"]["min_value"]

    If the section or key is absent, fallback defaults are used.
    """

    section = cfg.get(section_name, {})

    output = defaults.copy()

    if section:
        output.update(section)

    return output


def create_train_only_nest_emphasis_duplicates(train_df, cfg):
    """
    Create nest-emphasized duplicate images from the training split only.

    Why split first?
        We must avoid leakage where an original image lands in validation
        but a near-duplicate nest-emphasized version lands in training.

    Output:
        A dataframe with the same essential columns as train_df:
            image_path, score

        These rows point to generated PNG files inside:
            results/nest_emphasis_train_aug/
    """

    nest_cfg = get_nested_config(
        cfg,
        "nest_emphasis",
        {
            "enabled": True,
            "min_value": 145,
            "max_saturation": 115,
            "max_lab_b": 150,
            "dim_factor": 0.30,
            "blur_background": True,
            "blur_kernel": 31
        }
    )

    if not nest_cfg.get("enabled", True):
        print("Nest-emphasis augmentation disabled.")
        return pd.DataFrame(columns=train_df.columns)

    aug_dir = cfg["output_dir"] / "nest_emphasis_train_aug"
    os.makedirs(aug_dir, exist_ok=True)

    augmented_rows = []

    print(f"\nCreating train-only nest-emphasized duplicates in:\n  {aug_dir}")

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Creating nest-emphasis training images"):
        original_path = Path(row["image_path"])

        pil_img = load_image(original_path)

        if pil_img is None:
            print(f"Skipping unreadable image: {original_path}")
            continue

        emphasized_img, _ = create_nest_emphasized_image(
            pil_img,
            min_value=nest_cfg["min_value"],
            max_saturation=nest_cfg["max_saturation"],
            max_lab_b=nest_cfg["max_lab_b"],
            dim_factor=nest_cfg["dim_factor"],
            blur_background=nest_cfg["blur_background"],
            blur_kernel=nest_cfg["blur_kernel"]
        )

        # Include parent folder and row index to avoid filename collisions
        # across different source datasets that may reuse image names.
        source_dir = row["source_dir"] if "source_dir" in train_df.columns else original_path.parent.name

        safe_stem = f"{source_dir}_{original_path.stem}_row{idx}_nestemph"
        save_path = aug_dir / f"{safe_stem}.png"

        emphasized_img.save(save_path)

        new_row = row.copy()
        new_row["image_path"] = str(save_path)

        if "source_dir" in new_row.index:
            new_row["source_dir"] = f"{source_dir}_nestemph"

        if "filename" in new_row.index:
            new_row["filename"] = save_path.name

        augmented_rows.append(new_row)

    if not augmented_rows:
        print("No nest-emphasized duplicates were created.")
        return pd.DataFrame(columns=train_df.columns)

    aug_df = pd.DataFrame(augmented_rows)

    print(f"Created {len(aug_df)} nest-emphasized training duplicates.")

    return aug_df


def train_model(cfg, model_idx, train_loader, val_loader, device):
    model = create_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_loss = float("inf")

    patience_counter = 0

    os.makedirs(cfg["model_dir"], exist_ok=True)

    best_path = cfg["model_dir"] / f"model_{model_idx}_best.pt"

    print(f"Saving best model {model_idx} to:\n  {best_path}")

    for epoch in range(cfg["epochs"]):
        model.train()

        train_loss = 0.0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device).float()

            optimizer.zero_grad()

            outputs = model(imgs).squeeze()

            loss = tukey_biweight_loss(outputs, lbls)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation uses original, non-nest-emphasized validation images only.
        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device).float()

                outputs = model(imgs).squeeze()

                loss = tukey_biweight_loss(outputs, lbls)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Model {model_idx} | Epoch {epoch + 1}/{cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, path=best_path)
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= cfg["early_stopping_patience"]:
                print(f"Model {model_idx}: early stopping triggered at epoch {epoch + 1}.")
                break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to project config.yaml"
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model directory: {cfg['model_dir']}")

    # ---------------------------------------------------------
    # LOAD ORIGINAL LABELED DATA
    # ---------------------------------------------------------

    df = load_labels(cfg["scored_dir"])

    print(f"Loaded {len(df)} original labeled images.")

    # ---------------------------------------------------------
    # SPLIT ORIGINAL IMAGES FIRST
    # ---------------------------------------------------------
    # This is critical. We split before generating nest-emphasis duplicates
    # so augmented twins cannot leak across train/validation.

    train_df, val_df = train_test_split(
        df,
        test_size=1.0 - cfg["train_val_split"],
        random_state=42
    )

    print(f"Original train images: {len(train_df)}")
    print(f"Original validation images: {len(val_df)}")

    # ---------------------------------------------------------
    # CREATE TRAIN-ONLY NEST-EMPHASIS DUPLICATES
    # ---------------------------------------------------------

    aug_df = create_train_only_nest_emphasis_duplicates(train_df, cfg)

    if len(aug_df) > 0:
        train_df_expanded = pd.concat([train_df, aug_df], ignore_index=True)
    else:
        train_df_expanded = train_df.reset_index(drop=True)

    print(f"Expanded training images: {len(train_df_expanded)}")
    print(f"Validation images remain original-only: {len(val_df)}")

    # ---------------------------------------------------------
    # TRANSFORMS
    # ---------------------------------------------------------

    random_erasing_cfg = get_nested_config(
        cfg,
        "random_erasing",
        {
            "enabled": True,
            "p": 0.20,
            "scale_min": 0.01,
            "scale_max": 0.05,
            "ratio_min": 0.3,
            "ratio_max": 3.3
        }
    )

    train_transform = get_transforms(
        image_size=cfg["image_size"],
        mode="train",
        use_random_erasing=random_erasing_cfg.get("enabled", True),
        random_erasing_p=random_erasing_cfg.get("p", 0.20),
        random_erasing_scale=(
            random_erasing_cfg.get("scale_min", 0.01),
            random_erasing_cfg.get("scale_max", 0.05)
        ),
        random_erasing_ratio=(
            random_erasing_cfg.get("ratio_min", 0.3),
            random_erasing_cfg.get("ratio_max", 3.3)
        )
    )

    val_transform = get_transforms(
        image_size=cfg["image_size"],
        mode="val"
    )

    train_loader = DataLoader(
        NestDataset(train_df_expanded, train_transform),
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        NestDataset(val_df, val_transform),
        batch_size=cfg["batch_size"],
        shuffle=False
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---------------------------------------------------------
    # TRAIN ENSEMBLE
    # ---------------------------------------------------------

    for i in range(cfg["ensemble_size"]):
        print(f"\n--- Training Ensemble Member {i} ---")
        train_model(cfg, i, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
# EoF (train.py)