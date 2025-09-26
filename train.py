import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.annotations["label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.class_to_idx[self.annotations.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)
        return image, label

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data",
                        help="Path to dataset folder (expects images/ + labels.csv)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=str, default="../results")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Load dataset ---
    train_csv = os.path.join(args.data, "train_labels.csv")
    val_csv   = os.path.join(args.data, "val_labels.csv")
    train_dir = os.path.join(args.data, "images")
    val_dir   = os.path.join(args.data, "images")

    train_dataset = CustomImageDataset(train_csv, train_dir, transform=transform)
    val_dataset   = CustomImageDataset(val_csv, val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dat
