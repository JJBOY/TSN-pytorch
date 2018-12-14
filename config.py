import argparse
parser=argparse.ArgumentParser(
        description="Pytorch impelementation of  Temporal Segment Networks"
        )

parser.add_argument('--dataset',type=str,choices=['ucf101','hmdb51','kinetics'],default='ucf101')
parser.add_argument('--modality',type=str,choices=['RGB','RGBDiff','Flow'],default='RGB')
parser.add_argument('--train_list',type=str,default='./raw/train_list.txt')
parser.add_argument('--val_list',type=str,default='./raw/test_list.txt')
parser.add_argument('--root_path',type=str,default='./raw/data/')

#=======================model config==========================#
parser.add_argument('--arch',type=str,default='resnet101')
parser.add_argument('--num_segments',type=int,default=3)
parser.add_argument('--consensus_type',type=str,default='avg',
                    choices=['avg','max','topk','identity','rnn','cnn']
                    )
parser.add_argument('--k',type=int,default=3)
parser.add_argument('--dropout',type=float,default=0.5)


#=======================learning config==========================#
parser.add_argument('--epochs',default=45,type=int)
parser.add_argument('--batch_size',default=256,type=int)
parser.add_argument('--lr',default=0.001,type=float)
parser.add_argument('--clip-gradient', default=None, type=float)
parser.add_argument('--partial_bn', default=True, action="store_false")
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)

# ========================= Runtime Configs ==========================
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--evaluate', default=False, action='store_true')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--gpus',nargs='+',type=int, default=1)
parser.add_argument('--flow_prefix', default="", type=str)