from Diet_model import *
from Generate_Dieted_model import *

import os
import torch
import torch.nn as nn
from torchsummary import summary

MODEL_PATH = './output_Diet_ReLU/best_unet_model.pth'

#Hyperparameter
n_channels = 4 # 입력 채널 수
n_classes = 3  # 출력 클래스 수
bilinear = True #업샘플링 방법
learning_rate = 0.001  # 학습률
batch_size = 4  # 배치 크기
num_epochs = 80

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear).to(device)

print(model)