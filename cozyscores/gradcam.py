# gradcam.py - Produce heatmap overlay of pixel contribution
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from cozyscores.utils import (
    create_model,
    load_model,
    load_image,
    get_transforms,
    load_unscored_images,
    load_config,
    GradCAM
)

def overlay_heatmap(img_np, heatmap):
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.65, colored, 0.35, 0)
    return overlay

def generate_gradcams(cfg, mode="single"):
    """
    Generate Grad-CAM visualizations for CozyScores predictions.

    Modes:
    -------
    "single"
        Uses only the first ensemble member.
        Fastest and simplest. Good for quick inspection.

    "average"
        Averages Grad-CAM heatmaps across all ensemble members.
        Produces cleaner and more stable manuscript-quality figures.

    "panel"
        Generates a side-by-side comparison showing each ensemble
        member independently. Useful for debugging disagreement or
        shortcut learning behavior.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}") # debugging print statements
    print(f"Model directory resolved to: {cfg['model_dir']}")
    print(f"Heatmap directory resolved to: {cfg['heatmap_dir']}")

    # Create separate folders depending on Grad-CAM mode
    output_dir = Path(cfg["heatmap_dir"]) / mode
    os.makedirs(output_dir, exist_ok=True)

    transform = get_transforms(cfg["image_size"], mode="val")

    # ---------------------------------------------------------
    # LOAD ENSEMBLE MODELS
    # ---------------------------------------------------------

    models = []
    for i in range(cfg["ensemble_size"]):
        path = cfg["model_dir"] / f"model_{i}_best.pt"
        print(f"Loading Grad-CAM model {i} from: {path}") # debugging print statement

        if not path.exists():
            raise FileNotFoundError(f"Missing ensemble model: {path}")
    
        model = create_model()
        model = load_model(model, None, path, device)[0]
        model.eval()
        models.append(model)

    image_paths = load_unscored_images(cfg["unscored_dir"])

    # ---------------------------------------------------------
    # PROCESS EACH IMAGE
    # ---------------------------------------------------------

    for img_path in tqdm(image_paths, desc=f"Generating GradCAMs ({mode})"):
        pil_img = load_image(img_path)
        if pil_img is None:
            continue

        # Convert PIL image to numpy array for OpenCV overlay operations
        img_np = np.array(pil_img)

        # Shape becomes:
        # [1, channels, height, width]
        #
        # unsqueeze(0) adds batch dimension because PyTorch models
        # always expect batches, even for single images.
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        save_name = Path(img_path).stem + "_gradcam.png"
        save_path = output_dir / save_name

        # =====================================================
        # MODE 1: SINGLE MODEL
        # =====================================================

        if mode == "single":
            # Uses ONLY first ensemble member
            # Assumes ensemble members learned similar representations
            model = models[0]
            target_layer = model.layer4[-1]
            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate(input_tensor)
            overlay = overlay_heatmap(img_np, heatmap)
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        # =====================================================
        # MODE 2: AVERAGED ENSEMBLE HEATMAP
        # =====================================================

        elif mode == "average":
            heatmaps = []
            # Generate Grad-CAM from every ensemble member
            for model in models:
                target_layer = model.layer4[-1]
                gradcam = GradCAM(model, target_layer)
                heatmap = gradcam.generate(input_tensor)
                heatmaps.append(heatmap)

            # Average all heatmaps together pixel-by-pixel
            avg_heatmap = np.mean(heatmaps, axis=0)
            overlay = overlay_heatmap(img_np, avg_heatmap)
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        # =====================================================
        # MODE 3: SIDE-BY-SIDE PANEL
        # =====================================================

        elif mode == "panel":
            n_models = len(models)
            fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

            # If ensemble size = 1, matplotlib returns non-list axis
            if n_models == 1:
                axes = [axes]

            for idx, model in enumerate(models):
                target_layer = model.layer4[-1]
                gradcam = GradCAM(model, target_layer)
                heatmap = gradcam.generate(input_tensor)
                overlay = overlay_heatmap(img_np, heatmap)
                axes[idx].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(f"Model {idx}")
                axes[idx].axis("off")

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        else:
            raise ValueError(
                f"Unknown GradCAM mode: {mode}. "
                f"Choose from: single, average, panel"
            )

    print(f"Saved Grad-CAM images to: {output_dir}")

def main():

    parser = argparse.ArgumentParser()
    
    # Define command-line arguments
    parser.add_argument(
        "--config",
        required=True,
        help="Path to project config.yaml"
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "average", "panel"],
        help="GradCAM visualization mode"
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    generate_gradcams(cfg, mode=args.mode) # Use modes "single", "average", "panel"

if __name__ == "__main__":
    main()
# EoF gradcam.py