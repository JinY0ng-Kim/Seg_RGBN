# Seg_RGBN
Crop/Weed Segmentation using U-net with RGB+NIR dataset
* A 1x1 Convolution is added after two 3x3 Convolutions in the U-net architecture
* The NIR channel is concatenated to the RGB channels, resulting in a 4-channel input, which is used as the model input
* Input is resized from 1294x964 to 640x480.

# Dataset
Sunflower Dataset http://www.diag.uniroma1.it/~labrococo/fsd/sunflowerdatasets.html

# Experiments Result
<p align="center">
  <img src="https://github.com/user-attachments/assets/77501c93-228c-4995-bb67-56d7f70914b6" width="330" />
  <img src="https://github.com/user-attachments/assets/ec2aa1a8-46fd-434a-b87f-7728cc5e017a" width="242" />
  <img src="https://github.com/user-attachments/assets/7b21f156-c0ee-48c4-807e-e63a60ca148e" width="404" />
</p>

# Environment
RTX 3080 Ti 8GB, Cuda 12.2, Python 3.9.0, torch 2.4.0, torchvision 0.19.0
