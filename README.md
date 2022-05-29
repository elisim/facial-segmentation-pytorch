# Facial Segmentation in PyTorch

In this exercise you will build and train a face parts segmentation network from scratch.

## Dataset

**link:** [https://github.com/massimomauro/FASSEG-repository/tree/master/V2](https://github.com/massimomauro/FASSEG-repository/tree/master/V2)

**Please use the data from the V2 folder**, but feel free to split the train/test as you see fit.
**Note**: the github repo is just for the data, you may disregard all the rest.

## Guidelines

You are not expected to achieve state of the art results (or near that).
Network should be simple enough to train on the CPU with reasonable time (upto 15 min).

You may use numpy ,matplotlib and python builtins. (+library of your choice to read images).
**Make sure you write the network and training loop from scratch using PyTorch
building blocks (i.e. conv/relu/bn).**

## How To Submit

If possible use a jupyter notebook, to show code + code “walk-through” + results + explanations.
**Make sure you explain your architecture and main blocks in the training scheme. 
Also elaborate on the results with analysis.**


## Download dataset 
```bash
>>> bash download_dataset.sh
```

## Code Structure
```
│   .gitignore
│   Dataset-Visualization.ipynb --- quick data visualization
│   download_dataset.sh --- script downloading the dataset
│   Example-Solution.ipynb --- load data, train the model and show results
│   README.md
│
├───src
│       face_segmentation_dataset.py --- PyTorch dataset implementation with transformations
│       models.py --- segmentation models to solve the task
│       utils.py --- utils functions, e.g. converting the mask to labelled tensor
│       __init__.py
│
└───tests
        1.bmp
        Test-Mask-To-Label.ipynb 
```

