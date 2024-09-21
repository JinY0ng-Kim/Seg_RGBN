import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 클래스별 색상 값을 인덱스로 매핑하는 딕셔너리 예제 (필요에 따라 수정)
color_to_class = {
    (0, 0, 0): 0,      # 배경 (검정색)
    (0, 255, 0): 1,    # 클래스 1 (초록색)
    (0, 0, 255): 2,    # 클래스 2 (빨강색)
}

def rgb_to_class(mask, color_to_class):
    height, width, _ = mask.shape
    class_mask = np.zeros((height, width), dtype=np.uint8)
    for color, class_idx in color_to_class.items():
        mask_color = np.all(mask == np.array(color), axis=-1)
        class_mask[mask_color] = class_idx
    return class_mask

def load_data(image_path, nir_path, mask_path):
    images = []
    masks = []
    
    img_name = os.listdir(image_path)
    for img_name in tqdm(img_name, desc="Loading data"):
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
    
    return images, masks