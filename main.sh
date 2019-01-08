#!/bin/bash
export PYTHONUNBUFFERED="True"
python main.py --dataset ucf101 \
               --modality RGB \
               --interval 1 \
               --evaluate \
               --resume ./record/RGB/RGBbest.pth \
               --On_Video True


