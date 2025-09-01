import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = r"D:\projects\major-project\Brain_tumor_project"
DATA_DIR = os.path.join(BASE_DIR, "dataset_split")

# ---------------------------
# Custom Dataset Class (2D)
# ---------------------------
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.data_dir, "images")
        self.mask_dir = os.path.join(self.data_dir, "masks")
        self.transform = transform

        # Match patients with _images.npy
        self.patients = [
            f.replace("_images.npy", "")
            for f in os.listdir(self.image_dir)
            if f.endswith("_images.npy")
        ]

        self.data = []
        for patient in self.patients:
            img_path = os.path.join(self.image_dir, f"{patient}_images.npy")
            mask_path = os.path.join(self.mask_dir, f"{patient}_mask.npy")

            if not os.path.exists(mask_path):
                continue

            imgs = np.load(img_path)   # (H, W, slices, 1)
            masks = np.load(mask_path) # (H, W, slices)

            for i in range(imgs.shape[2]):  # iterate slices
                self.data.append((patient, i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient, slice_idx = self.data[idx]

        img_path = os.path.join(self.image_dir, f"{patient}_images.npy")
        mask_path = os.path.join(self.mask_dir, f"{patient}_mask.npy")

        image = np.load(img_path)[:, :, slice_idx, 0]  # shape (H, W)
        mask = np.load(mask_path)[:, :, slice_idx]     # shape (H, W)

        # Normalize image
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # Add channel dim
        image = np.expand_dims(image, axis=0)  # (1, H, W)
        mask = np.expand_dims(mask, axis=0)    # (1, H, W)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long), patient

# ---------------------------
# Load Dataset
# ---------------------------
train_dataset = BrainTumorDataset(DATA_DIR, split="train")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Debug prints
print("✅ Total training dataset size (slices):", len(train_dataset))

for images, masks, patients in train_loader:
    print("Batch patients:", patients)
    print("Image batch shape:", images.shape)  # (B, 1, H, W)
    print("Mask batch shape:", masks.shape)    # (B, 1, H, W)
    break
