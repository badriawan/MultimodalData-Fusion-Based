import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class FusionDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        self.img_size = img_size

        # ✅ FILTER: hanya folder
        self.samples = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.samples)

    def _find_file(self, folder, keyword):
        for f in os.listdir(folder):
            if keyword in f.lower():
                return os.path.join(folder, f)
        return None

    def __getitem__(self, idx):
        sample_folder = os.path.join(self.root_dir, self.samples[idx])

        depth_path = self._find_file(sample_folder, "depth")
        therm_path = self._find_file(sample_folder, "therm")
        rgb_path   = self._find_file(sample_folder, ".png")

        if depth_path is None or therm_path is None or rgb_path is None:
            raise FileNotFoundError(f"Missing modality in {sample_folder}")

        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        therm = cv2.imread(therm_path, cv2.IMREAD_GRAYSCALE)
        rgb   = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)

        if depth is None or therm is None or rgb is None:
            raise ValueError(f"Unreadable image in {sample_folder}")

        depth = cv2.resize(depth, (self.img_size, self.img_size))
        therm = cv2.resize(therm, (self.img_size, self.img_size))
        rgb   = cv2.resize(rgb, (self.img_size, self.img_size))

        depth = torch.tensor(depth, dtype=torch.float32) / 255.0
        therm = torch.tensor(therm, dtype=torch.float32) / 255.0
        rgb   = torch.tensor(rgb, dtype=torch.float32) / 255.0

        input_tensor = torch.stack([depth, therm, rgb], dim=0)
        target = rgb.unsqueeze(0)

        return input_tensor, target

from torch.utils.data import DataLoader, random_split


data_directory = "/content/drive/MyDrive/S3 UTP/MS2_dataset/fusion_dataset"

dataset = FusionDataset(data_directory)
print("Total valid samples:", len(dataset))

x, y = dataset[0]
print(x.shape, y.shape)

train_size = int(0.8 * len(dataset))  # 200
test_size = len(dataset) - train_size  # 50

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

import torch.nn as nn

class CNNFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNNFusion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

epochs = 50

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss/len(train_loader):.5f}")


from skimage.metrics import structural_similarity as ssim

def dice_score(pred, gt, threshold=0.5):
    pred = (pred > threshold).float()
    gt = (gt > threshold).float()
    intersection = (pred * gt).sum()
    return (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)

model.eval()
ssim_scores = []
dice_scores = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        pred = outputs.squeeze().cpu().numpy()
        gt = targets.squeeze().cpu().numpy()

        ssim_scores.append(ssim(pred, gt, data_range=1.0))
        dice_scores.append(dice_score(outputs, targets).item())

print("Average SSIM:", np.mean(ssim_scores))
print("Average Dice Score:", np.mean(dice_scores))

#Display Fused Sample
import matplotlib.pyplot as plt
model.eval()

with torch.no_grad():
    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)

    fused = model(inputs)

fused_img = fused.squeeze().cpu().numpy()

plt.figure(figsize=(5,5))
plt.imshow(fused_img, cmap='gray')
plt.title("Fused Image")
plt.axis("off")
plt.show()
