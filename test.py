from Diet_model import *
#from Generate_Dieted_model import *

import os
import torch
import torch.nn as nn
from torchsummary import summary

OUTPUT_DIR = './test'
MODEL_PATH = './output_Diet_ReLU/best_unet_model.pth'

# output 폴더 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
model.load_state_dict(torch.load(MODEL_PATH))


new_model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear).to(device)


summary(new_model, input_size=(n_channels, 480, 640))

# 가중치 검증
mapping = {
    # inc layer
    'inc.double_conv.0.weight': 'inc.double_conv.0.weight',
    'inc.double_conv.0.bias': 'inc.double_conv.0.bias',
    'inc.double_conv.1.weight': 'inc.double_conv.1.weight',
    'inc.double_conv.1.bias': 'inc.double_conv.1.bias',
    'inc.double_conv.1.running_mean': 'inc.double_conv.1.running_mean',
    'inc.double_conv.1.running_var': 'inc.double_conv.1.running_var',
    'inc.double_conv.1.num_batches_tracked': 'inc.double_conv.1.num_batches_tracked',
    'inc.double_conv.3.weight': 'inc.double_conv.3.weight',
    'inc.double_conv.3.bias': 'inc.double_conv.3.bias',
    'inc.double_conv.4.weight': 'inc.double_conv.4.weight',
    'inc.double_conv.4.bias': 'inc.double_conv.4.bias',
    'inc.double_conv.4.running_mean': 'inc.double_conv.4.running_mean',
    'inc.double_conv.4.running_var': 'inc.double_conv.4.running_var',
    'inc.double_conv.4.num_batches_tracked': 'inc.double_conv.4.num_batches_tracked',
    'inc.double_conv.6.weight': 'inc.double_conv.6.weight',
    'inc.double_conv.6.bias': 'inc.double_conv.6.bias',
    'inc.double_conv.7.weight': 'inc.double_conv.7.weight',
    'inc.double_conv.7.bias': 'inc.double_conv.7.bias',
    'inc.double_conv.7.running_mean': 'inc.double_conv.7.running_mean',
    'inc.double_conv.7.running_var': 'inc.double_conv.7.running_var',
    'inc.double_conv.7.num_batches_tracked': 'inc.double_conv.7.num_batches_tracked',

    # down1 layer
    'down1.maxpool_conv.1.double_conv.0.weight': 'down1.maxpool_conv.1.double_conv.0.weight',
    'down1.maxpool_conv.1.double_conv.0.bias': 'down1.maxpool_conv.1.double_conv.0.bias',
    'down1.maxpool_conv.1.double_conv.1.weight': 'down1.maxpool_conv.1.double_conv.1.weight',
    'down1.maxpool_conv.1.double_conv.1.bias': 'down1.maxpool_conv.1.double_conv.1.bias',
    'down1.maxpool_conv.1.double_conv.1.running_mean': 'down1.maxpool_conv.1.double_conv.1.running_mean',
    'down1.maxpool_conv.1.double_conv.1.running_var': 'down1.maxpool_conv.1.double_conv.1.running_var',
    'down1.maxpool_conv.1.double_conv.1.num_batches_tracked': 'down1.maxpool_conv.1.double_conv.1.num_batches_tracked',
    'down1.maxpool_conv.1.double_conv.3.weight': 'down1.maxpool_conv.1.double_conv.3.weight',
    'down1.maxpool_conv.1.double_conv.3.bias': 'down1.maxpool_conv.1.double_conv.3.bias',
    'down1.maxpool_conv.1.double_conv.4.weight': 'down1.maxpool_conv.1.double_conv.4.weight',
    'down1.maxpool_conv.1.double_conv.4.bias': 'down1.maxpool_conv.1.double_conv.4.bias',
    'down1.maxpool_conv.1.double_conv.4.running_mean': 'down1.maxpool_conv.1.double_conv.4.running_mean',
    'down1.maxpool_conv.1.double_conv.4.running_var': 'down1.maxpool_conv.1.double_conv.4.running_var',
    'down1.maxpool_conv.1.double_conv.4.num_batches_tracked': 'down1.maxpool_conv.1.double_conv.4.num_batches_tracked',
    'down1.maxpool_conv.1.double_conv.6.weight': 'down1.maxpool_conv.1.double_conv.6.weight',
    'down1.maxpool_conv.1.double_conv.6.bias': 'down1.maxpool_conv.1.double_conv.6.bias',
    'down1.maxpool_conv.1.double_conv.7.weight': 'down1.maxpool_conv.1.double_conv.7.weight',
    'down1.maxpool_conv.1.double_conv.7.bias': 'down1.maxpool_conv.1.double_conv.7.bias',
    'down1.maxpool_conv.1.double_conv.7.running_mean': 'down1.maxpool_conv.1.double_conv.7.running_mean',
    'down1.maxpool_conv.1.double_conv.7.running_var': 'down1.maxpool_conv.1.double_conv.7.running_var',
    'down1.maxpool_conv.1.double_conv.7.num_batches_tracked': 'down1.maxpool_conv.1.double_conv.7.num_batches_tracked',

    # down2 layer
    'down2.maxpool_conv.1.double_conv.0.weight': 'down2.maxpool_conv.1.double_conv.0.weight',
    'down2.maxpool_conv.1.double_conv.0.bias': 'down2.maxpool_conv.1.double_conv.0.bias',
    'down2.maxpool_conv.1.double_conv.1.weight': 'down2.maxpool_conv.1.double_conv.1.weight',
    'down2.maxpool_conv.1.double_conv.1.bias': 'down2.maxpool_conv.1.double_conv.1.bias',
    'down2.maxpool_conv.1.double_conv.1.running_mean': 'down2.maxpool_conv.1.double_conv.1.running_mean',
    'down2.maxpool_conv.1.double_conv.1.running_var': 'down2.maxpool_conv.1.double_conv.1.running_var',
    'down2.maxpool_conv.1.double_conv.1.num_batches_tracked': 'down2.maxpool_conv.1.double_conv.1.num_batches_tracked',
    'down2.maxpool_conv.1.double_conv.3.weight': 'down2.maxpool_conv.1.double_conv.3.weight',
    'down2.maxpool_conv.1.double_conv.3.bias': 'down2.maxpool_conv.1.double_conv.3.bias',
    'down2.maxpool_conv.1.double_conv.4.weight': 'down2.maxpool_conv.1.double_conv.4.weight',
    'down2.maxpool_conv.1.double_conv.4.bias': 'down2.maxpool_conv.1.double_conv.4.bias',
    'down2.maxpool_conv.1.double_conv.4.running_mean': 'down2.maxpool_conv.1.double_conv.4.running_mean',
    'down2.maxpool_conv.1.double_conv.4.running_var': 'down2.maxpool_conv.1.double_conv.4.running_var',
    'down2.maxpool_conv.1.double_conv.4.num_batches_tracked': 'down2.maxpool_conv.1.double_conv.4.num_batches_tracked',
    'down2.maxpool_conv.1.double_conv.6.weight': 'down2.maxpool_conv.1.double_conv.6.weight',
    'down2.maxpool_conv.1.double_conv.6.bias': 'down2.maxpool_conv.1.double_conv.6.bias',
    'down2.maxpool_conv.1.double_conv.7.weight': 'down2.maxpool_conv.1.double_conv.7.weight',
    'down2.maxpool_conv.1.double_conv.7.bias': 'down2.maxpool_conv.1.double_conv.7.bias',
    'down2.maxpool_conv.1.double_conv.7.running_mean': 'down2.maxpool_conv.1.double_conv.7.running_mean',
    'down2.maxpool_conv.1.double_conv.7.running_var': 'down2.maxpool_conv.1.double_conv.7.running_var',
    'down2.maxpool_conv.1.double_conv.7.num_batches_tracked': 'down2.maxpool_conv.1.double_conv.7.num_batches_tracked',

    # down3 layer
    'down3.maxpool_conv.1.double_conv.0.weight': 'down3.maxpool_conv.1.double_conv.0.weight',
    'down3.maxpool_conv.1.double_conv.0.bias': 'down3.maxpool_conv.1.double_conv.0.bias',
    'down3.maxpool_conv.1.double_conv.1.weight': 'down3.maxpool_conv.1.double_conv.1.weight',
    'down3.maxpool_conv.1.double_conv.1.bias': 'down3.maxpool_conv.1.double_conv.1.bias',
    'down3.maxpool_conv.1.double_conv.1.running_mean': 'down3.maxpool_conv.1.double_conv.1.running_mean',
    'down3.maxpool_conv.1.double_conv.1.running_var': 'down3.maxpool_conv.1.double_conv.1.running_var',
    'down3.maxpool_conv.1.double_conv.1.num_batches_tracked': 'down3.maxpool_conv.1.double_conv.1.num_batches_tracked',
    'down3.maxpool_conv.1.double_conv.3.weight': 'down3.maxpool_conv.1.double_conv.3.weight',
    'down3.maxpool_conv.1.double_conv.3.bias': 'down3.maxpool_conv.1.double_conv.3.bias',
    'down3.maxpool_conv.1.double_conv.4.weight': 'down3.maxpool_conv.1.double_conv.4.weight',
    'down3.maxpool_conv.1.double_conv.4.bias': 'down3.maxpool_conv.1.double_conv.4.bias',
    'down3.maxpool_conv.1.double_conv.4.running_mean': 'down3.maxpool_conv.1.double_conv.4.running_mean',
    'down3.maxpool_conv.1.double_conv.4.running_var': 'down3.maxpool_conv.1.double_conv.4.running_var',
    'down3.maxpool_conv.1.double_conv.4.num_batches_tracked': 'down3.maxpool_conv.1.double_conv.4.num_batches_tracked',
    'down3.maxpool_conv.1.double_conv.6.weight': 'down3.maxpool_conv.1.double_conv.6.weight',
    'down3.maxpool_conv.1.double_conv.6.bias': 'down3.maxpool_conv.1.double_conv.6.bias',
    'down3.maxpool_conv.1.double_conv.7.weight': 'down3.maxpool_conv.1.double_conv.7.weight',
    'down3.maxpool_conv.1.double_conv.7.bias': 'down3.maxpool_conv.1.double_conv.7.bias',
    'down3.maxpool_conv.1.double_conv.7.running_mean': 'down3.maxpool_conv.1.double_conv.7.running_mean',
    'down3.maxpool_conv.1.double_conv.7.running_var': 'down3.maxpool_conv.1.double_conv.7.running_var',
    'down3.maxpool_conv.1.double_conv.7.num_batches_tracked': 'down3.maxpool_conv.1.double_conv.7.num_batches_tracked',

    # down4 layer
    'down4.maxpool_conv.1.double_conv.0.weight': 'down4.maxpool_conv.1.double_conv.0.weight',
    'down4.maxpool_conv.1.double_conv.0.bias': 'down4.maxpool_conv.1.double_conv.0.bias',
    'down4.maxpool_conv.1.double_conv.1.weight': 'down4.maxpool_conv.1.double_conv.1.weight',
    'down4.maxpool_conv.1.double_conv.1.bias': 'down4.maxpool_conv.1.double_conv.1.bias',
    'down4.maxpool_conv.1.double_conv.1.running_mean': 'down4.maxpool_conv.1.double_conv.1.running_mean',
    'down4.maxpool_conv.1.double_conv.1.running_var': 'down4.maxpool_conv.1.double_conv.1.running_var',
    'down4.maxpool_conv.1.double_conv.1.num_batches_tracked': 'down4.maxpool_conv.1.double_conv.1.num_batches_tracked',
    'down4.maxpool_conv.1.double_conv.3.weight': 'down4.maxpool_conv.1.double_conv.3.weight',
    'down4.maxpool_conv.1.double_conv.3.bias': 'down4.maxpool_conv.1.double_conv.3.bias',
    'down4.maxpool_conv.1.double_conv.4.weight': 'down4.maxpool_conv.1.double_conv.4.weight',
    'down4.maxpool_conv.1.double_conv.4.bias': 'down4.maxpool_conv.1.double_conv.4.bias',
    'down4.maxpool_conv.1.double_conv.4.running_mean': 'down4.maxpool_conv.1.double_conv.4.running_mean',
    'down4.maxpool_conv.1.double_conv.4.running_var': 'down4.maxpool_conv.1.double_conv.4.running_var',
    'down4.maxpool_conv.1.double_conv.4.num_batches_tracked': 'down4.maxpool_conv.1.double_conv.4.num_batches_tracked',
    'down4.maxpool_conv.1.double_conv.6.weight': 'down4.maxpool_conv.1.double_conv.6.weight',
    'down4.maxpool_conv.1.double_conv.6.bias': 'down4.maxpool_conv.1.double_conv.6.bias',
    'down4.maxpool_conv.1.double_conv.7.weight': 'down4.maxpool_conv.1.double_conv.7.weight',
    'down4.maxpool_conv.1.double_conv.7.bias': 'down4.maxpool_conv.1.double_conv.7.bias',
    'down4.maxpool_conv.1.double_conv.7.running_mean': 'down4.maxpool_conv.1.double_conv.7.running_mean',
    'down4.maxpool_conv.1.double_conv.7.running_var': 'down4.maxpool_conv.1.double_conv.7.running_var',
    'down4.maxpool_conv.1.double_conv.7.num_batches_tracked': 'down4.maxpool_conv.1.double_conv.7.num_batches_tracked',

    # up1 layer
    'up1.conv.double_conv.0.weight': 'up1.conv.double_conv.0.weight',
    'up1.conv.double_conv.0.bias': 'up1.conv.double_conv.0.bias',
    'up1.conv.double_conv.1.weight': 'up1.conv.double_conv.1.weight',
    'up1.conv.double_conv.1.bias': 'up1.conv.double_conv.1.bias',
    'up1.conv.double_conv.1.running_mean': 'up1.conv.double_conv.1.running_mean',
    'up1.conv.double_conv.1.running_var': 'up1.conv.double_conv.1.running_var',
    'up1.conv.double_conv.1.num_batches_tracked': 'up1.conv.double_conv.1.num_batches_tracked',
    'up1.conv.double_conv.3.weight': 'up1.conv.double_conv.3.weight',
    'up1.conv.double_conv.3.bias': 'up1.conv.double_conv.3.bias',
    'up1.conv.double_conv.4.weight': 'up1.conv.double_conv.4.weight',
    'up1.conv.double_conv.4.bias': 'up1.conv.double_conv.4.bias',
    'up1.conv.double_conv.4.running_mean': 'up1.conv.double_conv.4.running_mean',
    'up1.conv.double_conv.4.running_var': 'up1.conv.double_conv.4.running_var',
    'up1.conv.double_conv.4.num_batches_tracked': 'up1.conv.double_conv.4.num_batches_tracked',
    'up1.conv.double_conv.6.weight': 'up1.conv.double_conv.6.weight',
    'up1.conv.double_conv.6.bias': 'up1.conv.double_conv.6.bias',
    'up1.conv.double_conv.7.weight': 'up1.conv.double_conv.7.weight',
    'up1.conv.double_conv.7.bias': 'up1.conv.double_conv.7.bias',
    'up1.conv.double_conv.7.running_mean': 'up1.conv.double_conv.7.running_mean',
    'up1.conv.double_conv.7.running_var': 'up1.conv.double_conv.7.running_var',
    'up1.conv.double_conv.7.num_batches_tracked': 'up1.conv.double_conv.7.num_batches_tracked',

    # up2 layer
    'up2.conv.double_conv.0.weight': 'up2.conv.double_conv.0.weight',
    'up2.conv.double_conv.0.bias': 'up2.conv.double_conv.0.bias',
    'up2.conv.double_conv.1.weight': 'up2.conv.double_conv.1.weight',
    'up2.conv.double_conv.1.bias': 'up2.conv.double_conv.1.bias',
    'up2.conv.double_conv.1.running_mean': 'up2.conv.double_conv.1.running_mean',
    'up2.conv.double_conv.1.running_var': 'up2.conv.double_conv.1.running_var',
    'up2.conv.double_conv.1.num_batches_tracked': 'up2.conv.double_conv.1.num_batches_tracked',
    'up2.conv.double_conv.3.weight': 'up2.conv.double_conv.3.weight',
    'up2.conv.double_conv.3.bias': 'up2.conv.double_conv.3.bias',
    'up2.conv.double_conv.4.weight': 'up2.conv.double_conv.4.weight',
    'up2.conv.double_conv.4.bias': 'up2.conv.double_conv.4.bias',
    'up2.conv.double_conv.4.running_mean': 'up2.conv.double_conv.4.running_mean',
    'up2.conv.double_conv.4.running_var': 'up2.conv.double_conv.4.running_var',
    'up2.conv.double_conv.4.num_batches_tracked': 'up2.conv.double_conv.4.num_batches_tracked',
    'up2.conv.double_conv.6.weight': 'up2.conv.double_conv.6.weight',
    'up2.conv.double_conv.6.bias': 'up2.conv.double_conv.6.bias',
    'up2.conv.double_conv.7.weight': 'up2.conv.double_conv.7.weight',
    'up2.conv.double_conv.7.bias': 'up2.conv.double_conv.7.bias',
    'up2.conv.double_conv.7.running_mean': 'up2.conv.double_conv.7.running_mean',
    'up2.conv.double_conv.7.running_var': 'up2.conv.double_conv.7.running_var',
    'up2.conv.double_conv.7.num_batches_tracked': 'up2.conv.double_conv.7.num_batches_tracked',

    # up3 layer
    'up3.conv.double_conv.0.weight': 'up3.conv.double_conv.0.weight',
    'up3.conv.double_conv.0.bias': 'up3.conv.double_conv.0.bias',
    'up3.conv.double_conv.1.weight': 'up3.conv.double_conv.1.weight',
    'up3.conv.double_conv.1.bias': 'up3.conv.double_conv.1.bias',
    'up3.conv.double_conv.1.running_mean': 'up3.conv.double_conv.1.running_mean',
    'up3.conv.double_conv.1.running_var': 'up3.conv.double_conv.1.running_var',
    'up3.conv.double_conv.1.num_batches_tracked': 'up3.conv.double_conv.1.num_batches_tracked',
    'up3.conv.double_conv.3.weight': 'up3.conv.double_conv.3.weight',
    'up3.conv.double_conv.3.bias': 'up3.conv.double_conv.3.bias',
    'up3.conv.double_conv.4.weight': 'up3.conv.double_conv.4.weight',
    'up3.conv.double_conv.4.bias': 'up3.conv.double_conv.4.bias',
    'up3.conv.double_conv.4.running_mean': 'up3.conv.double_conv.4.running_mean',
    'up3.conv.double_conv.4.running_var': 'up3.conv.double_conv.4.running_var',
    'up3.conv.double_conv.4.num_batches_tracked': 'up3.conv.double_conv.4.num_batches_tracked',
    'up3.conv.double_conv.6.weight': 'up3.conv.double_conv.6.weight',
    'up3.conv.double_conv.6.bias': 'up3.conv.double_conv.6.bias',
    'up3.conv.double_conv.7.weight': 'up3.conv.double_conv.7.weight',
    'up3.conv.double_conv.7.bias': 'up3.conv.double_conv.7.bias',
    'up3.conv.double_conv.7.running_mean': 'up3.conv.double_conv.7.running_mean',
    'up3.conv.double_conv.7.running_var': 'up3.conv.double_conv.7.running_var',
    'up3.conv.double_conv.7.num_batches_tracked': 'up3.conv.double_conv.7.num_batches_tracked',

    # up4 layer
    'up4.conv.double_conv.0.weight': 'up4.conv.double_conv.0.weight',
    'up4.conv.double_conv.0.bias': 'up4.conv.double_conv.0.bias',
    'up4.conv.double_conv.1.weight': 'up4.conv.double_conv.1.weight',
    'up4.conv.double_conv.1.bias': 'up4.conv.double_conv.1.bias',
    'up4.conv.double_conv.1.running_mean': 'up4.conv.double_conv.1.running_mean',
    'up4.conv.double_conv.1.running_var': 'up4.conv.double_conv.1.running_var',
    'up4.conv.double_conv.1.num_batches_tracked': 'up4.conv.double_conv.1.num_batches_tracked',
    'up4.conv.double_conv.3.weight': 'up4.conv.double_conv.3.weight',
    'up4.conv.double_conv.3.bias': 'up4.conv.double_conv.3.bias',
    'up4.conv.double_conv.4.weight': 'up4.conv.double_conv.4.weight',
    'up4.conv.double_conv.4.bias': 'up4.conv.double_conv.4.bias',
    'up4.conv.double_conv.4.running_mean': 'up4.conv.double_conv.4.running_mean',
    'up4.conv.double_conv.4.running_var': 'up4.conv.double_conv.4.running_var',
    'up4.conv.double_conv.4.num_batches_tracked': 'up4.conv.double_conv.4.num_batches_tracked',
    'up4.conv.double_conv.6.weight': 'up4.conv.double_conv.6.weight',
    'up4.conv.double_conv.6.bias': 'up4.conv.double_conv.6.bias',
    'up4.conv.double_conv.7.weight': 'up4.conv.double_conv.7.weight',
    'up4.conv.double_conv.7.bias': 'up4.conv.double_conv.7.bias',
    'up4.conv.double_conv.7.running_mean': 'up4.conv.double_conv.7.running_mean',
    'up4.conv.double_conv.7.running_var': 'up4.conv.double_conv.7.running_var',
    'up4.conv.double_conv.7.num_batches_tracked': 'up4.conv.double_conv.7.num_batches_tracked',

    # outc layer
    'outc.conv.weight': 'outc.conv.weight',
    'outc.conv.bias': 'outc.conv.bias',
}


# 가중치 복사 함수 정의
def copy_weights(old_model, new_model):
    old_dict = old_model.state_dict()
    new_dict = new_model.state_dict()

    # 가중치 복사
    for old_key, new_key in mapping.items():
        if old_key in old_dict and new_key in new_dict:
            new_dict[new_key] = old_dict[old_key]

    new_model.load_state_dict(new_dict)
    return new_model

new_model = copy_weights(model, new_model)

torch.save(new_model.state_dict(), os.path.join(OUTPUT_DIR, 'test.pth'))


# 가중치 검증 함수 정의
def verify_weights(old_model, new_model, mapping):
    old_dict = old_model.state_dict()
    new_dict = new_model.state_dict()

    for old_key, new_key in mapping.items():
        if old_key in old_dict and new_key in new_dict:
            old_weight = old_dict[old_key]
            new_weight = new_dict[new_key]
            if torch.equal(old_weight, new_weight):
                print(f"Weights for {old_key} -> {new_key} copied successfully.")
            else:
                print(f"Weights for {old_key} -> {new_key} do not match.")

verify_weights(model, new_model, mapping)

