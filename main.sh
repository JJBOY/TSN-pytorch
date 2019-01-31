#!/bin/bash
export PYTHONUNBUFFERED="True"
python main.py --dataset sthsth \
               --modality RGBDiff \
               --interval 2 \
               --On_Video True \
               --train_list /home/qx/project/data/sthsth/train.txt \
               --val_list /home/qx/project/data/sthsth/test.txt \
               --root_path /home/qx/project/data/sthsth/data/ \
               --batch_size 20 \


