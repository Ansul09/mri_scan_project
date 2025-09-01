# train_unet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader_2d import BrainTumorDataset
from unet_model import UNet

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
DATA_DIR = r"D:\projects\major-project\Brain_tumor_project\dataset_split"
train_dataset = BrainTumorDataset(DATA_DIR, split="train")
val_dataset = BrainTumorDataset(DATA_DIR, split="val")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Model
model = UNet(in_channels=1, out_channels=1).to(DEVICE)

# Loss (Dice + BCEWithLogits)
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1e-6):
    # Apply sigmoid to convert logits -> probabilities
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    return 1 - ((2. * intersection + smooth) / (denom + smooth)).mean()

def combined_loss(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
EPOCHS = 1
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_loader:
        imgs, masks = batch[0], batch[1]   # ignore patient_id
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).float()  # ✅ FIX

        preds = model(imgs)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs, masks = batch[0], batch[1]
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).float()  # ✅ FIX
            preds = model(imgs)
            val_loss += combined_loss(preds, masks).item()

    print(f"Epoch {epoch+1}/{EPOCHS}, "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}")
