#!/bin/bash
export PYTHONUNBUFFERED="True"
python main.py --dataset 'ucf101' \
               --modality 'RGB'\
               --train_list './raw/train_list.txt'\
               --val_list './raw/test_list.txt'\
               --root_path '../C3D/raw/data/'

