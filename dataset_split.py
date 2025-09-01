import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Match patients by file name prefix
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(".npy")])

        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = np.load(img_path)  # (H, W, 1) or (H, W)
        mask = np.load(mask_path)  # (H, W, 1) or (H, W)

        # Convert to float32/int
        image = image.astype(np.float32)
        mask = mask.astype(np.int64)

        # Ensure correct shape for images: (C, H, W)
        if image.ndim == 3:  # (H, W, 1)
            image = np.transpose(image, (2, 0, 1))  # -> (1, H, W)
        elif image.ndim == 2:  # (H, W)
            image = np.expand_dims(image, axis=0)   # -> (1, H, W)

        # Ensure masks are 2D: (H, W)
        if mask.ndim == 3 and mask.shape[-1] == 1:  # (H, W, 1)
            mask = np.squeeze(mask, axis=-1)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask



# 🔹 Example usage
if __name__ == "__main__":
    train_images = r"D:\projects\major-project\Brain_tumor_project\dataset_split\train\images"
    train_masks = r"D:\projects\major-project\Brain_tumor_project\dataset_split\train\masks"

    dataset = BrainTumorDataset(train_images, train_masks)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Debug: Check one batch
    for images, masks in dataloader:
        print("Image batch shape:", images.shape)  # (B, C, H, W)
        print("Mask batch shape:", masks.shape)    # (B, H, W)
        break
