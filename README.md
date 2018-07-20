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

## Run NetDissect in PyTorch

* You can configure `settings.py` to load your own model, or change the default parameters.

* Run NetDissect 

```
    python main.py
```


## NetDissect Result

* At the end of the dissection script, a report will be generated inside `result` folder that summarizes the interpretable units of the tested network. These are, respectively, the HTML-formatted report, the semantics of the units of the layer summarized as a bar graph, visualizations of all the units of the layer (using zero-indexed unit numbers), and a CSV file containing raw scores of the top matching semantic concepts in each category for each unit of the layer.


## Reference
If you find the codes useful, please cite this paper
```
@inproceedings{netdissect2017,
  title={Network Dissection: Quantifying Interpretability of Deep Visual Representations},
  author={Bau, David and Zhou, Bolei and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}
```

