# Facial Segmentation in PyTorch

Very simple solution for FASSEG dataset, using UNet ad base model.

## Dataset

**link:** [https://github.com/massimomauro/FASSEG-repository/tree/master/V2](https://github.com/massimomauro/FASSEG-repository/tree/master/V2)

**Please use the data from the V2 folder**, but feel free to split the train/test as you see fit.
**Note**: the github repo is just for the data, you may disregard all the rest.

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

