# nest_emphasis_qc.py
# Generate QC panels for automated nest-emphasis preprocessing.

import os
import sys
from pathlib import Path

# Allow imports from the CozyScores root.
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cozyscores.utils import (
    load_config,
    load_labels,
    load_image,
    create_nest_emphasized_image
)


def sample_across_scores(df, n=30, seed=42):
    """
    Try to sample images across the 1-5 score range rather than taking
    a purely random sample dominated by overrepresented score 4 images.
    """

    df = df.copy()

    # Approximate score bin: 1, 2, 3, 4, or 5.
    df["score_bin"] = df["score"].round().clip(1, 5).astype(int)

    rng = np.random.default_rng(seed)

    sampled_parts = []
    per_bin = max(1, n // 5)

    for score_bin in [1, 2, 3, 4, 5]:
        sub = df[df["score_bin"] == score_bin]

        if len(sub) == 0:
            continue

        take_n = min(per_bin, len(sub))
        sampled_parts.append(sub.sample(n=take_n, random_state=seed + score_bin))

    sampled = (
        sampled_parts[0]
        if len(sampled_parts) == 1
        else np.concatenate([part.index.values for part in sampled_parts])
    )

    if not isinstance(sampled, np.ndarray):
        selected = sampled_parts[0]
    else:
        selected = df.loc[sampled]

    # Fill remaining slots randomly from the images not already selected.
    if len(selected) < n:
        remaining = df.drop(index=selected.index, errors="ignore")
        if len(remaining) > 0:
            fill_n = min(n - len(selected), len(remaining))
            fill = remaining.sample(n=fill_n, random_state=seed)
            selected = selected._append(fill, ignore_index=False)

    return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_qc_panel(
    pil_img,
    emphasized_img,
    mask,
    save_path,
    title=""
):
    """
    Save a 3-panel QC image:
    Original | Detected likely paper mask | Nest-emphasized image
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(pil_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Likely Paper Mask")
    axes[1].axis("off")

    axes[2].imshow(emphasized_img)
    axes[2].set_title("Nest-Emphasized")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to project config.yaml"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of QC examples to generate"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for QC image sampling"
    )

    parser.add_argument(
        "--min-value",
        type=int,
        default=145,
        help="HSV brightness threshold for likely paper pixels"
    )

    parser.add_argument(
        "--max-saturation",
        type=int,
        default=115,
        help="HSV saturation threshold for likely paper pixels"
    )

    parser.add_argument(
        "--dim-factor",
        type=float,
        default=0.30,
        help="Brightness multiplier for non-paper regions"
    )

    parser.add_argument(
        "--no-blur",
        action="store_true",
        help="Disable background blur; only dim non-paper regions"
    )

    parser.add_argument(
        "--max-lab-b",
        type=int,
        default=150,
        help="LAB b-channel threshold; lower values reject more yellow bedding/sticky note"
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    df = load_labels(cfg["scored_dir"])

    if len(df) == 0:
        raise RuntimeError(f"No scored images found in {cfg['scored_dir']}")

    selected = sample_across_scores(df, n=args.n, seed=args.seed)

    qc_dir = cfg["output_dir"] / "nest_emphasis_qc"
    os.makedirs(qc_dir, exist_ok=True)

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Generating nest-emphasis QC"):
        img_path = row["image_path"]
        score = row["score"]

        pil_img = load_image(img_path)

        if pil_img is None:
            continue

        emphasized_img, mask = create_nest_emphasized_image(
            pil_img,
            min_value=args.min_value,
            max_saturation=args.max_saturation,
            max_lab_b=args.max_lab_b,
            dim_factor=args.dim_factor,
            blur_background=not args.no_blur
        )

        img_path_obj = Path(img_path)

        # Include parent folder so duplicate filenames from different sessions do not overwrite each other.
        safe_name = f"{img_path_obj.parent.name}_{img_path_obj.stem}_score_{score:.3f}_qc.png"
        save_path = qc_dir / safe_name

        title = f"{img_path_obj.parent.name} | {img_path_obj.name} | score={score:.3f}"

        save_qc_panel(
            pil_img=pil_img,
            emphasized_img=emphasized_img,
            mask=mask,
            save_path=save_path,
            title=title
        )

    print(f"Saved QC panels to: {qc_dir}")


if __name__ == "__main__":
    main()