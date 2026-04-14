#!/usr/bin/env bash

for i in {0..9}; do
    python3 src/modelpipeline/inference.py --video "data/FaceForensics/original/00$i.mp4"
done

for i in {10..99}; do
    python3 src/modelpipeline/inference.py --video "data/FaceForensics/original/0$i.mp4"
done
