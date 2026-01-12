import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class FusionDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        self.samples = sorted(os.listdir(root_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.samples[idx])

        depth = cv2.imread(os.path.join(sample_path, f"{self.samples[idx]}_depth.png"),
                           cv2.IMREAD_GRAYSCALE)
        therm = cv2.imread(os.path.join(sample_path, f"{self.samples[idx]}_therm.png"),
                           cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(os.path.join(sample_path, f"{self.samples[idx]}.png"),
                         cv2.IMREAD_GRAYSCALE)

        depth = cv2.resize(depth, (self.img_size, self.img_size))
        therm = cv2.resize(therm, (self.img_size, self.img_size))
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))

        depth = torch.tensor(depth, dtype=torch.float32) / 255.0
        therm = torch.tensor(therm, dtype=torch.float32) / 255.0
        rgb = torch.tensor(rgb, dtype=torch.float32) / 255.0

        input_tensor = torch.stack([depth, therm, rgb], dim=0)
        target = rgb.unsqueeze(0)  # pseudo GT

        return input_tensor, target


from torch.utils.data import DataLoader, random_split

dataset = FusionDataset("fusion_dataset")

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
