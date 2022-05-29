# Facial Segmentation in PyTorch
Exercise description: [PyTorch from scratch - Face Parts Segmentation](https://www.notion.so/beyondminds/PyTorch-from-scratch-Face-Parts-Segmentation-9f49d7471b68499aaac6f4037ebc2dff)

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



