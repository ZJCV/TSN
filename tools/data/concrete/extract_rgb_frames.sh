#!/usr/bin/env bash

cd ../
python build_rawframes.py ../../data/concrete/videos/ ../../data/concrete/rawframes/ --task rgb --level 2  --ext mp4
echo "Genearte raw frames (RGB only)"

cd concrete/
