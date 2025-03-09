## Setting up the dataset
---------------------------------------------

1.  Open a terminal or command prompt on your operating system.

2.  Navigate to the directory where you cloned this repo.

3.  Run the below command, and press Enter to execute it.
```
pip install gdown
python setup-dataset.py
```

Two datasets will be downloaded in a `FaceMaskDataset` folder.
1. Detection and classification dataset will be in `dataset` folder.
2. Segmentation dataset will be in a zip file `MaskedFaceSegmentation.zip`, extract it manually.

Task wise folders are created, we can add our code into the respective folders.

