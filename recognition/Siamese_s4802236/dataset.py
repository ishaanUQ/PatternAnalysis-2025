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

# Processing the Metadata
metadata = path + "/train-metadata.csv"
df_raw = pd.read_csv(metadata, index_col=0)
df = df_raw[["isic_id", "target"]].copy()
df["isic_id"] = df["isic_id"].astype(str) + ".jpg"
df.rename(columns={"isic_id": "image_name"}, inplace=True)

# Copying images from cashe cleaned data directory
raw_images_path = root / "train-image" / "image"
data = Path("data")
cleaned_images = data / "train-image" / "image"
cleaned_images.mkdir(parents=True, exist_ok=True)

missing = 0
for image in df["image_name"]:
    src = raw_images_path / image
    dst = cleaned_images / image
    if src.exists():
        if not dst.exists():  # Skip if already copied
            shutil.copy2(src, dst)
    else:
        missing += 1
#Should know if any images are missing
if missing:
    print(f"{missing} images missing, check dataset and retry")
else: 
    print("All good")
    #store metadata in cleaned directory as well
    (df[["image_name", "target"]]).to_csv(data / "train-metadata.csv", index=False)
