# eecs206a_project
EECS 206A Final Project

## Requirements
```angular2html
pip install numpy, scipy, matplotlib, pytransform3d
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
To sample a pose in configuration space,
```angular2html
python sample_pose.py  # sample a pose using one of the datasets in data/
```

To train a neural network for sampling task space,
```angular2html
python prepare_endpose.py  # prepare training dataset using one of the datasets in data/
python interpolate_endpose.py  # train a multilayer perceptron
```