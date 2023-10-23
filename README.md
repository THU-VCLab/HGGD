# HGGD

Official code of paper [Efficient Heatmap-Guided 6-Dof Grasp Detection in Cluttered Scenes](https://ieeexplore.ieee.org/document/10168242)

### [Paper](https://ieeexplore.ieee.org/abstract/document/10168242/) | [Video](https://www.youtube.com/watch?v=V8gG1eHbrsU)

# Framework

![framework](./images/framework.jpg)

# Requirements

- Python >= 3.8
- PyTorch >= 1.10
- pytorch3d
- numpy==1.23.5
- pandas
- cupoch
- numba
- grasp_nms
- matplotlib
- open3d
- opencv-python
- scikit-image
- tensorboardX
- torchsummary
- tqdm
- transforms3d
- trimesh
- autolab_core
- cvxopt

## Installation

This code has been tested on Ubuntu20.04 with Cuda 11.1/11.3/11.6, Python3.8/3.9 and Pytorch 1.11.0/1.12.0.

Get the code.

```bash
git clone https://github.com/THU-VCLab/HGGD.git
```

Create new Conda environment.

```bash
conda create -n hggd python=3.8
cd HGGD
```

Please install [pytorch](https://pytorch.org/) and [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) manually.

```bash
# pytorch-1.11.0
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# pytorch3d
pip install fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

Install other packages via Pip.

```bas
pip install -r requirements.txt
```

# Usage

## Checkpoint

Checkpoints (realsense/kinect) can be downloaded from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e3edfc2c8b114513b7eb/)

## Preprocessed Dataset

Preprocessed datasets (realsense.7z/kinect.7z) can be downloaded from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e3edfc2c8b114513b7eb/)

Containing converted and refined grasp poses from each image in graspnet dataset

## Train

Training code will be released in future.

## Test

Download and unzip our preprocessed datasets (for convenience), you can also try removing unnecessary parts in our test code and directly reading images from the original graspnet dataset api.

Run test code (read rgb and depth image from graspnet dataset and eval grasps).

```bash
bash test_graspnet.sh
```

Attention: if you want to change camera, please remember to change `camera` in [config.py](./dataset/config.py)

Typical hyperparameters:

```python
center-num # sampled local center/region number, higher number means more regions&grasps, but gets slower speed, default: 48
grid-size # grid size for our grid-based center sampling, higher number means sparser centers, default: 8
ratio # grasp attributes prediction downsample ratio, default: 8
anchor-k # classification anchor number for grasp in-plane rotation, default: 6
anchor-w # regress anchor size for grasp width, default: 50
anchor-z # regress anchor size for grasp depth, default: 20
all-points-num # downsampled point cloud point number, default: 25600
group-num # local region point cloud point number, default: 512
local-k # grasp detection number in each local region, default: 10
scene-l & scene-r # scene range, train: 0~100, seen: 100~130, similar: 130~160, novel: 160~190
input-h & input-w # downsampled input image size, should be 640x360
local-thres & heatmap-thres # heatmap and grasp score filter threshold, set to 0.01 in our settings
dataset-path # our preprocessed dataset path (read grasp poses)
scene-path # original graspnet dataset path (read images)
num-workers # eval worker number
dump-dir # detected grasp poses dumped path (used in later evaluation)
```

## Demo

Run demo code (read rgb and depth image from file and get grasps).

```bash
bash demo.sh
```

Typical hyperparameters:

```python
center-num # sampled local center/region number, higher number means more regions&grasps, but gets slower speed, default: 48
grid-size # grid size for our grid-based center sampling, higher number means sparser centers, default: 8
all-points-num # downsampled point cloud point number, default: 25600
group-num # local region point cloud point number, default: 512
local-k # grasp detection number in each local region, default: 10
```

# Citation

Please cite our paper in your publications if it helps your research:

```
@article{chen2023efficient,
  title={Efficient Heatmap-Guided 6-Dof Grasp Detection in Cluttered Scenes},
  author={Chen, Siang and Tang, Wei and Xie, Pengwei and Yang, Wenming and Wang, Guijin},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```
