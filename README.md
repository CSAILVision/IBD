# IBD: Interpretable Basis Decomposition for Visual Explanation

## Introduction
This repository contains the demo code for the ECCV'18 paper "Interpretable Basis Decomposition for Visual Explanation".

## Download
* Clone the code of Network Dissection Lite from github
```
    git clone https://github.com/CSAILVision/IBD
    cd IBD
```
* Download the Broden dataset (~1GB space) and the example pretrained model. If you already download this, you can create a symbolic link to your original dataset.
```
    ./script/dlbroden.sh
    ./script/dlzoo.sh
```

Note that AlexNet models work with 227x227 image input, while VGG, ResNet, GoogLeNet works with 224x224 image input.

## Requirements

* Python Environments

```
    pip3 install numpy sklearn scipy scikit-image matplotlib easydict torch torchvision
```

Note: The repo was written by pytorch-0.3.1. ([PyTorch](http://pytorch.org/), [Torchvision](https://github.com/pytorch/vision)) 

## Run IBD in PyTorch

* You can configure `settings.py` to load your own model, or change the default parameters.

* Run IBD 

```
    python3 main.py
```


## IBD Result

* At the end of the dissection script, a HTML-formatted report will be generated inside `result` folder that summarizes the interpretable units of the tested network. 


## Reference
If you find the codes useful, please cite this paper
```
@inproceedings{IBD2018,
  title={Interpretable Basis Decomposition for Visual Explanation},
  author={Zhou, Bolei* and Sun, Yiyou* and Bau, David* and Torralba, Antonio},
  booktitle={European Conference on Computer Vision},
  year={2018}
}
```

