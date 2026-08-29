#!/bin/bash

# This script downloads, extracts, and cleans up the CIFAR-100 dataset.
# It's designed to be run within a Lightning Studio environment.

echo "Creating datasets directory..."
mkdir -p /content/datasets
echo "Directory /content/datasets created (or already exists)."

echo "Downloading CIFAR-100 dataset from Hugging Face..."
# The -O flag specifies the output filename for the downloaded file.
wget -O /content/datasets/cifar-100-python.tar.gz https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz
echo "Download complete. File saved as /content/datasets/cifar-100-python.tar.gz"

echo "Extracting the dataset..."
# The -C flag specifies the directory to extract the contents into.
tar -xf /content/datasets/cifar-100-python.tar.gz -C /content/datasets/
echo "Extraction complete. Dataset contents are in /content/datasets/"

echo "Cleaning up the downloaded archive..."
rm /content/datasets/cifar-100-python.tar.gz
echo "Archive file removed."

echo "CIFAR-100 dataset download and extraction process finished successfully!"