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
import time


# 데이터 경로 설정
IMAGE_PATH = '/home/coral/jy/U-net_NIR/dataset/rgb'
NIR_PATH = '/home/coral/jy/U-net_NIR/dataset/nir'
MASK_PATH = '/home/coral/jy/U-net_NIR/dataset/gt_color'
OUTPUT_DIR = './output/prediction'
MODEL_PATH = './output1/best_unet_model.pth'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

color_to_class = {
    (0, 0, 0): 0,     # 검정
    (0, 255, 0): 1,   # 녹색
    (0, 0, 255): 2    # 파랑
}

# RGB 마스크를 클래스 마스크로 변환
def rgb_to_class(mask, color_to_class):
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for rgb, cls in color_to_class.items():
        class_mask[(mask == rgb).all(axis=2)] = cls
    return class_mask

# 데이터 로드 함수
def load_data(image_path, nir_path, mask_path):
    images = []
    masks = []
    image_names = os.listdir(image_path)
    
    for img_name in tqdm(image_names, desc="Loading data"):
        img_path = os.path.join(image_path, img_name)
        nir_img_path = os.path.join(nir_path, img_name)
        mask_img_path = os.path.join(mask_path, img_name)
        
        img = cv2.imread(img_path)
        nir_img = cv2.imread(nir_img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_img_path)
        
        if img is None or nir_img is None or mask is None:
            continue
        
        img = cv2.resize(img, (640, 480))
        nir_img = cv2.resize(nir_img, (640, 480))
        
        img = np.dstack((img, nir_img))
        
        mask = cv2.resize(mask, (640, 480))
        mask = rgb_to_class(mask, color_to_class)
        
        images.append(img)
        masks.append(mask)
    
    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32)
    
    return images, masks, image_names

# 데이터 로드 및 확인
images, masks, image_names = load_data(IMAGE_PATH, NIR_PATH, MASK_PATH)

n_channels = 4  # 입력 채널 수
n_classes = len(np.unique(masks))  # 출력 클래스 수
bilinear = True  # 업샘플링 방법
batch_size = 4  # 배치 크기

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 커스텀 데이터셋 클래스
class SunflowerDataset(Dataset):
    def __init__(self, images, masks, image_names):
        self.images = images
        self.masks = masks
        self.image_names = image_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image_name = self.image_names[idx]

        # 텐서로 변환
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask, image_name

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

def mask_to_color(mask, color_map):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in color_map.items():
        color_mask[mask == cls] = np.array(color, dtype=np.uint8)
    return color_mask

color_map = {
    0: [0, 0, 0],     # 클래스 0의 색상 (검정)
    1: [0, 255, 0],   # 클래스 1의 색상 (녹색)
    2: [0, 0, 255]    # 클래스 2의 색상 (파랑)
}

dataset = SunflowerDataset(images, masks, image_names)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 평가 및 시간 측정
ious = []
start_time = time.time()

with torch.no_grad():
    for idx, (images, true_masks, image_names_batch) in enumerate(tqdm(data_loader)):
        images = images.to(device)
        true_masks = true_masks.to(device)

        outputs = model(images)
        iou = calculate_iou(outputs, true_masks, n_classes)
        ious.append(iou)

        pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
        true_masks = true_masks.cpu().numpy()

        for j in range(pred_masks.shape[0]):
            pred_mask = pred_masks[j]
            true_mask = true_masks[j]
            pred_mask_rgb = mask_to_color(pred_mask, color_map)
            true_mask_rgb = mask_to_color(true_mask, color_map)

            base_name = os.path.basename(image_names_batch[j])
            pred_save_name = f'pred_{base_name}'
            true_save_name = f'true_{base_name}'
            pred_save_path = os.path.join(OUTPUT_DIR, pred_save_name)
            true_save_path = os.path.join(OUTPUT_DIR, true_save_name)

            cv2.imwrite(pred_save_path, pred_mask_rgb)
            cv2.imwrite(true_save_path, true_mask_rgb)

end_time = time.time()
total_time = end_time - start_time

# 결과 출력
mean_iou = np.nanmean(ious)
print(f'Mean IoU: {mean_iou}')
print(f'Total evaluation time: {total_time} seconds')

# IoU 분포 확인 및 저장
plt.hist(ious, bins=20, edgecolor='black')
plt.xlabel('IoU')
plt.ylabel('Frequency')
plt.title('IoU Distribution')
plt.savefig(os.path.join(OUTPUT_DIR, 'iou_distribution.png'))
plt.show()
