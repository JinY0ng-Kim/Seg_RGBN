from unet_model import *
from unet_data import *

import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 데이터 경로 설정
IMAGE_PATH = '/home/coral/jy/U-net_NIR/dataset/rgb'
NIR_PATH = '/home/coral/jy/U-net_NIR/dataset/nir'
MASK_PATH = '/home/coral/jy/U-net_NIR/dataset/gt_color'
OUTPUT_DIR = './output_Original_300epoch'

# output 폴더 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#data load & check
images, masks = load_data(IMAGE_PATH, NIR_PATH, MASK_PATH)
print(f"Images shape: {images.shape}")
print(f"Masks shape: {masks.shape}")
print(f"Mask image dtype: {masks.dtype}")
print(f"Unique values in mask image: {np.unique(masks)}")
print(f"num-classes: {len(np.unique(masks))}")

#Hyperparameter
n_channels = 4 # 입력 채널 수
n_classes = len(np.unique(masks))  # 출력 클래스 수
bilinear = True #업샘플링 방법
learning_rate = 0.001  # 학습률
batch_size = 4  # 배치 크기
num_epochs = 300

# 커스텀 데이터셋 클래스
class SunflowerDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # 텐서로 변환
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask

# 데이터를 학습용과 검증용으로 분할
images_train, images_val, masks_train, masks_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# 학습 및 검증 데이터 로더 생성
train_dataset = SunflowerDataset(images_train, masks_train)
val_dataset = SunflowerDataset(images_val, masks_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#Model, Loss function, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define IoU metric
def calculate_iou(pred_mask, true_mask, num_classes):
    ious = []
    pred_mask = torch.argmax(pred_mask, dim=1)
    
    for cls in range(num_classes):
        pred_cls = pred_mask == cls
        true_cls = true_mask == cls
        
        if true_cls.sum() == 0:
            ious.append(float('nan'))
        else:
            intersection = (pred_cls & true_cls).sum().item()
            union = (pred_cls | true_cls).sum().item()
            ious.append(intersection / union)
    return np.nanmean(ious)

# Training loop
train_losses = []
val_losses = []
iou_scores = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update(1)
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Training loss: {avg_train_loss}")

    model.eval()
    val_loss = 0
    iou_score = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            iou_score += calculate_iou(outputs, masks, n_classes)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_iou_score = iou_score / len(val_loader)
    val_losses.append(avg_val_loss)
    iou_scores.append(avg_iou_score)
    print(f"Epoch {epoch + 1}, Validation loss: {avg_val_loss}, IoU: {avg_iou_score}")

    # 모델 가중치 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_unet_model.pth'))

# Save the final model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_unet_model.pth'))

# 학습 결과 시각화
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, iou_scores, 'g-', label='IoU Score')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('IoU Score')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
plt.show()

"""
def mask_to_color(mask, color_map):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in color_map.items():
        color_mask[mask == cls] = np.array(color, dtype=np.uint8)
    return color_mask

# 예측 결과 저장 함수 수정
def save_predictions(images, masks, model, output_dir, color_map, n=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    with torch.no_grad():
        for i in range(n):
            idx = np.random.randint(0, len(images))
            img = torch.tensor(images[idx]).permute(2, 0, 1).unsqueeze(0).to(device)
            mask = masks[idx]

            # 예측 수행
            pred_mask = model(img)
            pred_mask = torch.argmax(pred_mask.squeeze(), dim=0).cpu().numpy()

            # 이미지를 3채널로 변환하여 저장
            img = img.squeeze().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            img = (img * 255).astype(np.uint8)  # [0, 1] -> [0, 255]
            cv2.imwrite(os.path.join(output_dir, f'input_{i}.png'), img)

            # 컬러 마스크 저장
            true_mask_color = mask_to_color(mask, color_map)
            pred_mask_color = mask_to_color(pred_mask, color_map)
            cv2.imwrite(os.path.join(output_dir, f'true_mask_{i}.png'), true_mask_color)
            cv2.imwrite(os.path.join(output_dir, f'pred_mask_{i}.png'), pred_mask_color)

# 예측 결과 저장
color_map = {
    0: [0, 0, 0],     # 클래스 0의 색상 (검정)
    1: [0, 255, 0],   # 클래스 1의 색상 (녹색)
    2: [0, 0, 255]    # 클래스 2의 색상 (파랑)
}

save_predictions(images_val, masks_val, model, OUTPUT_DIR, color_map)
"""