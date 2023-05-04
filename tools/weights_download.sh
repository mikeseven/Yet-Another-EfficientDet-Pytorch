#!/bin/bash

weights_root_url="https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download"
weights_local_dir="./weights"

mkdir -p $weights_local_dir

# original weights (v1.0)
wget "$weights_root_url/1.0/efficientdet-d0.pth" -P "$weights_local_dir"
wget "$weights_root_url/1.0/efficientdet-d1.pth" -P "$weights_local_dir"
wget "$weights_root_url/1.0/efficientdet-d2.pth" -P "$weights_local_dir"
wget "$weights_root_url/1.0/efficientdet-d3.pth" -P "$weights_local_dir"
wget "$weights_root_url/1.0/efficientdet-d4.pth" -P "$weights_local_dir"
wget "$weights_root_url/1.0/efficientdet-d5.pth" -P "$weights_local_dir"
wget "$weights_root_url/1.0/efficientdet-d6.pth" -P "$weights_local_dir"

# newer weights (v1.2)
wget "$weights_root_url/1.2/efficientdet-d7.pth" -P "$weights_local_dir"
wget "$weights_root_url/1.2/efficientdet-d8.pth" -P "$weights_local_dir"
