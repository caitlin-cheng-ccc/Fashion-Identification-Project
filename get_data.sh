#!/bin/bash
rm -rf data
rm -rf validate

# 1. Create train and tes directories
mkdir -p data/train
mkdir -p data/validate

# 2. Download Training Data
echo "Downloading training images"
wget -c https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip -O data/train2020.zip

# 3. Download Validation Data
echo "Downloading validation images"
wget -c https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip -O data/val_test2020.zip

# 4. Download Labels
echo "Downloading training labels"
wget -c https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json -O data/train/instances_attributes_train2020.json

echo "Downloading validation labels"
wget -c https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json -O data/validate/instances_attributes_val2020.json

# 5. Extract Images into SubfoldersA
echo "Extracting training images"
unzip -q data/train2020.zip -d data/train/images

echo "Extracting validation images"
unzip -q data/val_test2020.zip -d data/validate/images

# 6. Clean up temporary zip archives
rm data/train2020.zip data/val_test2020.zip

echo "Done"

