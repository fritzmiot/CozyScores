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

# Register HEIC support in Pillow
pillow_heif.register_heif_opener()

def load_config(project_config_path):
    """
    Loads the master config and merges it with the project-specific config.
    """
    # 1. Load Base Config (lives in the same folder as utils.py)
    base_config_path = Path(__file__).parent / "base_config.yaml"
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Project Config
    with open(project_config_path, 'r') as f:
        project_config = yaml.safe_load(f)

    # 3. Merge (Project overrides Base)
    if project_config:
        config.update(project_config)

    # 4. Path Resolution Magic
    # We make all paths absolute based on the Project Folder's location
    project_root = Path(project_config_path).parent
    
    path_keys = ["model_dir", "scored_dir", "unscored_dir", "output_dir"]
    for key in path_keys:
        # Resolve the path relative to the project_root
        config[key] = (project_root / config[key]).resolve()

    return config

def load_labels(scored_photos_dir) -> pd.DataFrame:
    all_data = []
    scored_photos_dir = Path(scored_photos_dir)
    
    if not scored_photos_dir.exists():
        raise FileNotFoundError(f"Directory not found: {scored_photos_dir}")

    subdirs = [d for d in scored_photos_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
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
    for dirpath, _, filenames in os.walk(unscored_dir):
        for fname in filenames:
            if fname.lower().endswith(valid_exts):
                image_paths.append(str(Path(dirpath) / fname))
    return image_paths

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

def get_transforms(image_size=1024, mode="train"):
    t = [LetterboxResize(size=(image_size, image_size))]
    if mode == "train":
        t.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(180)])
    t.append(transforms.ToTensor())
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

def save_model(model, optimizer=None, epoch=None, path=None):
    torch.save({'model_state_dict': model.state_dict()}, path)

def load_model(model, optimizer, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, optimizer, 0