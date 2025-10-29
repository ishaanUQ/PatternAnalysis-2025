# dataset.py

### IMPORTS ###
import os
import pandas as pd
import kagglehub as kh
from pathlib import Path
import shutil


# Authenticating Kaggle API
os.environ["KAGGLE_USERNAME"] = "ishaansood1"
os.environ["KAGGLE_KEY"] = "27613e6479fb99452da36eb26b5c5cb3"

# Downlaoding 224 x 224 dataset from Kaggle and metadata describing the images
# and storing in cashe
path = kh.dataset_download("nischaydnk/isic-2020-jpg-224x224-resized")
root = Path(path)