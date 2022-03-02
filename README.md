# ST-UNet
Pytorch codes for ['Swin Transformer Embedding UNet for Remote
Sensing Image Semantic Segmentation'](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9686686)

## Environment
conda create -n xxx python=3.7 -y

conda activate xxx

pip install -r requirements.txt

###requirements.txt
```bash
torchviz
torchvision==0.5.0
torch==1.4.0
timm==0.3.2
numpy
tqdm
tensorboard
tensorboardX
ml-collections
medpy
SimpleITK
scipy
h5py
opencv-python
pillow
toposort
tifffile
matplotlib
yacs 
einops
```



## Reference
* [TransUNet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)
* [SoftPool](https://github.com/alexandrosstergiou/SoftPool)
