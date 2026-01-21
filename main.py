import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

from Unet2encoders import UNetDualEncoderFusion


class FusionDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        self.img_size = img_size

        # Ambil semua file RGB (tanpa suffix)
        self.samples = sorted([
            f.replace(".png", "")
            for f in os.listdir(root_dir)
            if f.endswith(".png")
            and "_nir" not in f
            and "_therm" not in f
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        rgb_path   = os.path.join(self.root_dir, f"{sample_id}.png")
        nir_path   = os.path.join(self.root_dir, f"{sample_id}_nir.png")
        therm_path = os.path.join(self.root_dir, f"{sample_id}_therm.png")

        if not (os.path.exists(rgb_path) and os.path.exists(nir_path) and os.path.exists(therm_path)):
            raise FileNotFoundError(f"Missing modality for sample {sample_id}")

        rgb   = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        nir   = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        therm = cv2.imread(therm_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None or nir is None or therm is None:
            raise ValueError(f"Unreadable image for sample {sample_id}")

        rgb   = cv2.resize(rgb, (self.img_size, self.img_size))
        nir   = cv2.resize(nir, (self.img_size, self.img_size))
        therm = cv2.resize(therm, (self.img_size, self.img_size))

        rgb   = torch.tensor(rgb, dtype=torch.float32) / 255.0
        nir   = torch.tensor(nir, dtype=torch.float32) / 255.0
        therm = torch.tensor(therm, dtype=torch.float32) / 255.0

        # Input: (3, H, W) → [NIR, Thermal, RGB]
        input_tensor = torch.stack([nir, therm, rgb], dim=0)

        # Pseudo-GT (RGB)
        target = rgb.unsqueeze(0)

        return input_tensor, target


from torch.utils.data import DataLoader, random_split

data_directory = "/content/drive/MyDrive/S3 UTP/MS2_dataset/fusion_dataset_2"

dataset = FusionDataset(data_directory)
print("Total valid samples:", len(dataset))

x, y = dataset[0]
print("Input shape:", x.shape)
print("Target shape:", y.shape)

import torch

torch.manual_seed(42)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# split train → train + val
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

test_indices = test_dataset.indices
test_sample_names = [dataset.samples[i] for i in test_indices]

print("Test samples:")
for name in test_sample_names:
    print(name)


#CNN Model Based Fusion
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
print("Device using : ", device)


#fusion loss for NIR domination
# import torch.nn.functional as F

# def fusion_loss(fused, rgb, nir,
#                 w_rgb=0.4,
#                 w_nir=0.6):
#     loss_rgb = F.mse_loss(fused, rgb)
#     loss_nir = F.mse_loss(fused, nir)
#     return w_rgb * loss_rgb + w_nir * loss_nir



import torch.nn.functional as F

def gradient_loss(fused, nir):
    # horizontal gradient
    fx = fused[:, :, :, 1:] - fused[:, :, :, :-1]
    nx = nir[:, :, :, 1:] - nir[:, :, :, :-1]

    # vertical gradient
    fy = fused[:, :, 1:, :] - fused[:, :, :-1, :]
    ny = nir[:, :, 1:, :] - nir[:, :, :-1, :]

    return F.l1_loss(fx, nx) + F.l1_loss(fy, ny)



#Training Loop

# model = CNNFusion().to(device)

model = UNetDualEncoderFusion(use_attention=True).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    # ---------- TRAIN ----------
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)   # RGB pseudo-GT

        nir = inputs[:, 0:1, :, :]     # NIR channel

        rgb = targets                      # RGB pseudo-GT

       


        optimizer.zero_grad()
        outputs = model(inputs)

        # loss = fusion_loss(
        #     outputs,
        #     rgb=targets,
        #     nir=nir,
        #     w_rgb=0.4,
        #     w_nir=0.6
        # )

        loss = (
            0.5 * F.mse_loss(outputs, rgb) +
            0.3 * F.mse_loss(outputs, nir) +
            0.2 * gradient_loss(outputs, nir)
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            nir = inputs[:, 0:1, :, :]
            rgb = targets

            outputs = model(inputs)

            # loss = fusion_loss(
            #     outputs,
            #     rgb=targets,
            #     nir=nir,
            #     w_rgb=0.6,
            #     w_nir=0.4
            # )

            loss = (
            0.5 * F.mse_loss(outputs, rgb) +
            0.3 * F.mse_loss(outputs, nir) +
            0.2 * gradient_loss(outputs, nir)
        )

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {avg_train_loss:.5f} | "
        f"Val Loss: {avg_val_loss:.5f}"
    )


#save training plot
os.makedirs("training_logs", exist_ok=True)

plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Training Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("train_val_loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# # training selesai
# torch.save(model.state_dict(), "cnn_fusion_model_only.pth")

# model = CNNFusion().to(device)
# model.load_state_dict(torch.load("cnn_fusion_model_only.pth", map_location=device))
# model.eval()

# print("Model loaded successfully")


# training selesai
torch.save(model.state_dict(), "unet_dual_encoder_attention.pth")

model = UNetDualEncoderFusion(use_attention=True).to(device)
model.load_state_dict(
    torch.load("unet_dual_encoder_attention.pth", map_location=device)
)
model.eval()

print("Model loaded successfully")


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

#Refrences free-metrics
def entropy(img):
    img_uint8 = (img * 255).astype(np.uint8)
    hist = np.bincount(img_uint8.flatten(), minlength=256)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def mutual_information(img1, img2, bins=256):
    hgram, _, _ = np.histogram2d(
        img1.flatten(), img2.flatten(), bins=bins, range=[[0,1],[0,1]]
    )
    pxy = hgram / np.sum(hgram)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    return np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))

def edge_preservation(fused, img):
    fused_edge = cv2.Sobel(fused, cv2.CV_64F, 1, 1, ksize=3)
    img_edge   = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

    numerator = np.sum(fused_edge * img_edge)
    denominator = np.sqrt(
        np.sum(fused_edge ** 2) * np.sum(img_edge ** 2)
    ) + 1e-8

    return numerator / denominator

entropy_scores = []
mi_scores = []
epi_scores = []

model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        fused = model(inputs)

        fused_img = fused.squeeze().cpu().numpy()
        rgb = inputs[0,2].cpu().numpy()
        nir = inputs[0,0].cpu().numpy()
        therm = inputs[0,1].cpu().numpy()

        # Entropy
        entropy_scores.append(entropy(fused_img))

        # Mutual Information
        mi_total = (
            mutual_information(fused_img, rgb) +
            mutual_information(fused_img, nir) +
            mutual_information(fused_img, therm)
        )
        mi_scores.append(mi_total)

        # Edge Preservation
        epi = (
            edge_preservation(fused_img, rgb) +
            edge_preservation(fused_img, nir) +
            edge_preservation(fused_img, therm)
        ) / 3
        epi_scores.append(epi)

print("Average Entropy:", np.mean(entropy_scores))
print("Average Mutual Information:", np.mean(mi_scores))
print("Average Edge Preservation Index:", np.mean(epi_scores))


#Save Fused Sample
model.eval()

with torch.no_grad():
    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)

    fused = model(inputs)

fused_img = fused.squeeze().cpu().numpy()
fused_save = (fused_img * 255).astype("uint8")
cv2.imwrite("fusion_result.png", fused_save)


nir = inputs[0,0].cpu().numpy()
thermal = inputs[0,1].cpu().numpy()
rgb = inputs[0,2].cpu().numpy()

plt.figure(figsize=(12,3))

plt.subplot(1,4,1)
plt.imshow(nir, cmap='gray')
plt.title("nir")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(thermal, cmap='hot')
plt.title("thermal")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(rgb, cmap='gray')
plt.title("RGB")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(fused_img, cmap='gray')
plt.title("Fused Output")
plt.axis("off")

plt.tight_layout()
plt.savefig("fusion_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

#Original_Images
os.makedirs("original_images", exist_ok=True)

nir = inputs[0,0].cpu().numpy()
thermal = inputs[0,1].cpu().numpy()
rgb = inputs[0,2].cpu().numpy()

cv2.imwrite("original_images/nir.png", (nir * 255).astype("uint8"))
cv2.imwrite("original_images/thermal.png", (thermal * 255).astype("uint8"))
cv2.imwrite("original_images/rgb.png", (rgb * 255).astype("uint8"))


#Batch_Fused
os.makedirs("fusion_results/batch", exist_ok=True)

model.eval()
for i, (inputs, _) in enumerate(test_loader):
    inputs = inputs.to(device)

    with torch.no_grad():
        fused = model(inputs)

    fused_img = fused.squeeze().cpu().numpy()
    fused_save = (fused_img * 255).astype("uint8")

    cv2.imwrite(f"fusion_results/batch/fused_{i:03d}.png", fused_save)
