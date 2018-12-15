# TSN-pytorch
Pytorch impelementation for Temporal Segment Networks (TSN) in ECCV 2016

[原始版本实现](https://github.com/yjxiong/tsn-pytorch)，我只是稍微做了改动，自己训练了模型。

这个版本不需要光流，只用到了RGBDiff，需要训练光流的模态需要去使用[dense flow](https://github.com/yjxiong/dense_flow/tree/c9369a32ea491001db5298dfda1fa227a912d34f)提取光流。

## Dependency

[pytorch 0.4](https://pytorch.org/) 有无GPU都可以使用，但是无GPU可能会很慢。

[python3]()

[ffmpeg]() 可以直接 sudo apt-get install ffmpeg安装，用来把视频转换成图片存储，读写会快一些，[代码](https://github.com/JJBOY/TSN-pytorch/blob/master/raw/video2img.sh)，我使用的帧率是10FPS，为了节约存储空间的话可以调低一点。

[UCF101](http://crcv.ucf.edu/data/UCF101.php) 数据集，下好放在./raw/data/下面，如果外网比较慢的话，可以从我分享的[百度云](https://pan.baidu.com/s/1OlGZ5HSy63k8oj_bEcxowA)上下载。

##  Test on UCF101

训练和测试都只用到了split1，全都用上的应该能进一步提升。

注意：这里的效果比原论文要高不少，原因主要在于原论文使用的是BN-Inception，我使用的是ResNet101，但是我没在BN-Inception上测试。

| Modality | Top1 Accuracy   |  Top5 Accuracy      |
| :------: | :-------------: | :----: |
|   RGB    | 73.80% | 92.55% |
| RGBDiff  | 67.78% | 87.81% |
| Fusion   | 81.52% | 95.70% |

环境配好了之后需要下载我的预训练模型：

[RGB Model](https://pan.baidu.com/s/15TO-O4yo6Lljoh4sT0x9zA)

[RGBDiff Model](https://pan.baidu.com/s/1vQObFdfjMmb6hb78Feb6lw)

下载好了之后直接运行test.py ，python test.py 即可得到RGB和RGBDiff两个分支融合的结果。

也可以单独测试单个分支，python main.py --evaluate --resume [模型路径] --modality [RGB/RGBDiff]，或者直接去对应的修改一下[config.py](https://github.com/JJBOY/TSN-pytorch/blob/master/config.py)里面的参数。

## Train

只需要下载好对应的数据集，然后修改好合适的参数即可在自己的数据集上训练。

训练时间：UCF101 split1 ，GTX 2080Ti上大概两个小时。