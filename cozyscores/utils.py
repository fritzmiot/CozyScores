# utils.py - Utility functions for CozyScores project
import os
import yaml
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import pandas as pd
import numpy as np
from torchvision.models import ResNet18_Weights
import re
from pathlib import Path
import pillow_heif
import cv2
import matplotlib.cm as cm

# Register HEIC support in Pillow
pillow_heif.register_heif_opener()

def load_config(project_config_path):
    """
    Load base_config.yaml, merge project-specific overrides, and resolve paths.

    Important design choice:
        The user should only need to set scoring_model_version, e.g. "v4" or "v5".
        Paths such as model_dir and heatmap_dir are derived automatically so they
        cannot accidentally drift out of sync.

    Example:
        scoring_model_version: "v5"

    Derives:
        model_dir   = <project_root>/../../models/scoring/v5
        heatmap_dir = <project_root>/results/gradcam/scoring_v5
    """

    project_config_path = Path(project_config_path)
    project_root = project_config_path.parent

    # 1. Load base config from the same folder as utils.py
    base_config_path = Path(__file__).parent / "base_config.yaml"

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Load project-specific config
    with open(project_config_path, "r") as f:
        project_config = yaml.safe_load(f)

    # 3. Merge project config over base config
    if project_config:
        config.update(project_config)

    # ---------------------------------------------------------
    # Version-derived paths
    # ---------------------------------------------------------

    scoring_model_version = config.get("scoring_model_version", "v4")

    # Normalize in case someone writes 5 instead of "v5"
    scoring_model_version = str(scoring_model_version)

    if not scoring_model_version.startswith("v"):
        scoring_model_version = f"v{scoring_model_version}"

    config["scoring_model_version"] = scoring_model_version

    # Base directories resolved relative to the project config location.
    config["model_root"] = (project_root / config.get("model_root", "../../models/scoring")).resolve()
    config["data_dir"] = (project_root / config.get("data_dir", "data")).resolve()
    config["output_dir"] = (project_root / config.get("output_dir", "results")).resolve()

    # Standard input paths.
    config["scored_dir"] = (project_root / config.get("scored_dir", "data/scored_photos")).resolve()
    config["unscored_dir"] = (project_root / config.get("unscored_dir", "data/unscored_photos")).resolve()

    # Derived model path.
    config["model_dir"] = config["model_root"] / scoring_model_version

    # Derived versioned output paths.
    # Existing scripts can keep using heatmap_dir; future scripts can use the others.
    scoring_label = f"scoring_{scoring_model_version}"

    config["heatmap_dir"] = config["output_dir"] / "gradcam" / scoring_label
    config["prediction_output_dir"] = config["output_dir"] / "predictions" / scoring_label
    config["evaluation_output_dir"] = config["output_dir"] / "evaluation" / scoring_label
    config["qc_output_dir"] = config["output_dir"] / "segmentation_qc" / scoring_label

    # Future-proof paths for segmentation/morphometrics.
    config["metadata_dir"] = config["data_dir"] / "metadata"
    config["annotation_dir"] = config["data_dir"] / "segmentation_annotations"
    config["work_dir"] = project_root / "work"

    return config

def load_labels(scored_photos_dir) -> pd.DataFrame:
    all_data = []
    scored_photos_dir = Path(scored_photos_dir)
    
    if not scored_photos_dir.exists():
        raise FileNotFoundError(f"Directory not found: {scored_photos_dir}")

    subdirs = [d for d in scored_photos_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
        # Allows to set directory to be ignored by pipeline if file ".cozyscore_ignore" is present
        if (subdir / ".cozyscores_ignore").exists():
            print(f"Skipping ignored scored folder: {subdir}")
            continue

        csv_path = subdir / "scores.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df.columns = [col.lower().strip() for col in df.columns]
        image_col = "filename" if "filename" in df.columns else "image"
        score_col = "average_score" if "average_score" in df.columns else "score"

        df["image_path"] = df[image_col].apply(lambda fname: str((subdir / fname).resolve()))
        df["score"] = pd.to_numeric(df[score_col], errors="coerce")
        all_data.append(df[["image_path", "score"]])

    return pd.concat(all_data, ignore_index=True)

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def load_unscored_images(unscored_dir):
    valid_exts = (".jpg", ".jpeg", ".png", ".heic", ".HEIC")
    image_paths = []

    for dirpath, dirnames, filenames in os.walk(unscored_dir):
        dirpath_obj = Path(dirpath)

        if (dirpath_obj / ".cozyscores_ignore").exists():
            print(f"Skipping ignored unscored folder: {dirpath_obj}")
            dirnames[:] = []
            continue

        # Prevent os.walk from descending into ignored child folders.
        dirnames[:] = [
            d for d in dirnames
            if not (dirpath_obj / d / ".cozyscores_ignore").exists()
        ]

        for fname in filenames:
            if fname.lower().endswith(valid_exts):
                image_paths.append(str(dirpath_obj / fname))

    return image_paths

def make_nest_material_mask(
    pil_img,
    min_value=145,
    max_saturation=115,
    max_lab_b=150,
    min_component_area=25,
    dilate_iterations=2
    ):
    """
    Create a rough mask for likely white nesting material.

    This is not perfect segmentation. It is a fast heuristic designed to
    emphasize white paper strips while reducing attention to bedding,
    sticky notes, cage walls, and glare.

    Logic:
        1. HSV: paper is usually bright and relatively low saturation.
        2. LAB: paper is less yellow than corn-cob bedding or yellow sticky notes.
        3. Small connected components are removed to reduce bedding speckle/noise.

    Important:
        This function intentionally does NOT mask a fixed image corner.
        Some images may be rotated, and nests can occur near corners, so fixed
        spatial suppression could accidentally remove biologically relevant regions.

    Args:
        pil_img: RGB PIL image.
        min_value: HSV brightness threshold. Higher = stricter bright-pixel detection.
        max_saturation: HSV saturation threshold. Lower = stricter white/gray detection.
        max_lab_b: LAB yellow-blue threshold. Lower = rejects more yellow bedding/sticky note.
        min_component_area: Removes tiny specks/noise from the mask.
        dilate_iterations: Expands mask slightly so thin paper strips are preserved.

    Returns:
        mask: Boolean numpy array where True = likely nesting material.
    """

    img_rgb = np.array(pil_img.convert("RGB"))

    # HSV filter: likely paper is bright and not highly saturated.
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    hsv_mask = (v >= min_value) & (s <= max_saturation)

    # LAB filter: reject yellow-ish pixels.
    # In OpenCV LAB, b > ~128 trends yellow; neutral white/gray is closer to 128.
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    _, _, lab_b = cv2.split(lab)
    lab_mask = lab_b <= max_lab_b

    raw_mask = (hsv_mask & lab_mask).astype(np.uint8)

    # Remove small noise and close small holes.
    kernel = np.ones((3, 3), np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    # Remove tiny connected components that are likely bedding speckles or glare.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)

    clean_mask = np.zeros_like(raw_mask)

    for label_idx in range(1, num_labels):  # label 0 is background
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            clean_mask[labels == label_idx] = 1

    # Dilate so thin paper strips are not underrepresented.
    if dilate_iterations > 0:
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=dilate_iterations)

    return clean_mask.astype(bool)


def create_nest_emphasized_image(
    pil_img,
    min_value=145,
    max_saturation=115,
    max_lab_b=150,
    dim_factor=0.30,
    blur_background=True,
    blur_kernel=31
    ):
    """
    Create a nest-emphasized copy of an image.

    Likely nesting material is preserved at original brightness.
    Everything else is dimmed and optionally blurred.

    This is meant for training augmentation/QC, not as a final publication image.

    Returns:
        emphasized_img: PIL image with background de-emphasized.
        mask: Boolean numpy array of likely nesting-material pixels.
    """

    img_rgb = np.array(pil_img.convert("RGB"))

    mask = make_nest_material_mask(
        pil_img,
        min_value=min_value,
        max_saturation=max_saturation,
        max_lab_b=max_lab_b
    )

    if blur_background:
        # Gaussian blur kernels must be odd numbers.
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        background = cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), 0)
    else:
        background = img_rgb.copy()

    # Dim the non-mask region so paper remains the dominant visual signal.
    background = (background.astype(np.float32) * dim_factor).clip(0, 255).astype(np.uint8)

    emphasized = background.copy()

    # Preserve likely paper-strip pixels from the original image.
    emphasized[mask] = img_rgb[mask]

    emphasized_img = Image.fromarray(emphasized)

    return emphasized_img, mask


class ScaledSigmoid(torch.nn.Module):
    def __init__(self, min_val=1.0, max_val=5.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    def forward(self, x):
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(x)


class LetterboxResize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        original_width, original_height = img.size
        scale = min(self.size[0] / original_width, self.size[1] / original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        new_img = Image.new("RGB", self.size, (0, 0, 0))
        new_img.paste(img, ((self.size[0] - new_width) // 2, (self.size[1] - new_height) // 2))
        return new_img

def get_transforms(
    image_size=1024,
    mode="train",
    use_random_erasing=False,
    random_erasing_p=0.20,
    random_erasing_scale=(0.01, 0.05),
    random_erasing_ratio=(0.3, 3.3)
    ):
    """
    Create image transforms for CozyScores.

    Training mode:
        - Letterbox resize
        - Random flips/rotation
        - ToTensor
        - Optional mild RandomErasing

    Validation/prediction mode:
        - Letterbox resize
        - ToTensor only

    RandomErasing is placed after ToTensor because torchvision's
    RandomErasing operates on tensors, not PIL images.
    """

    t = [LetterboxResize(size=(image_size, image_size))]

    if mode == "train":
        t.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180)
        ])

    t.append(transforms.ToTensor())

    if mode == "train" and use_random_erasing:
        t.append(
            transforms.RandomErasing(
                p=random_erasing_p,
                scale=random_erasing_scale,
                ratio=random_erasing_ratio,
                value=0
            )
        )

    return transforms.Compose(t)


class NestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row['image_path'])
        if img is None: return self.__getitem__((idx + 1) % len(self.df))
        if self.transform: img = self.transform(img)
        return img, torch.tensor(float(row['score']), dtype=torch.float32)

def create_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 1),
        ScaledSigmoid(min_val=1.0, max_val=5.0)
    )
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad()

        output = self.model(input_tensor)

        output.backward(torch.ones_like(output))

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], device=input_tensor.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        if cam.max() != 0:
            cam /= cam.max()

        return cam.cpu().numpy()

def save_model(model, optimizer=None, epoch=None, path=None):
    torch.save({'model_state_dict': model.state_dict()}, path)

def load_model(model, optimizer, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, optimizer, 0
# EoF utils.py