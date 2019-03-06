from dataset.dataset import TSNDataSet
from model.model import TSN
from dataset.transforms import *
from config import parser
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset =='sthsth':
        num_class=174
    else:
        raise ValueError('Unknown dataset' + args.dataset)

    RGBmodel = TSN(num_class, args.num_segments, 'RGB',
                base_model=args.arch, consensus_type=args.consensus_type,
                dropout=args.dropout, partial_bn=not args.nopartial_bn).to(device)
    RGBDiffmodel = TSN(num_class, args.num_segments, 'RGBDiff',
               base_model=args.arch, consensus_type=args.consensus_type,
               dropout=args.dropout, partial_bn=not args.nopartial_bn).to(device)
    # RGBDiffmodel = TSN(num_class, args.num_segments, 'Flow',
    #             base_model=args.arch, consensus_type=args.consensus_type,
    #             dropout=args.dropout, partial_bn=not args.nopartial_bn).to(device)
    checkpoint = torch.load('./record/RGBbest.pth')
    RGBmodel.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load('./record/RGBDiffbest.pth')
    RGBDiffmodel.load_state_dict(checkpoint['state_dict'])
    # checkpoint = torch.load('./record/Flow/Flowbest.pth')
    # RGBDiffmodel.load_state_dict(checkpoint['state_dict'])

    crop_size = RGBmodel.crop_size
    scale_size = RGBmodel.scale_size
    input_mean = RGBmodel.input_mean
    input_std = RGBmodel.input_std

    RGB_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl="{:05d}.jpg" ,
                   random_shift=False,
                   test_mode=True,
                   On_Video=True,
                   interval=1,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(input_mean, input_std)
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    RGBDiff_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=5,
                   modality='RGBDiff',
                   image_tmpl="{:05d}.jpg" if args.modality in ["RGB","RGBDiff"]  else "{}/{}/frame{:06d}.jpg",
                   test_mode=True,
                   random_shift=False,
                   On_Video=True,
                   interval=2,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception')
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    #actually this is the Flow loader.i am lazy to change the name.
    # RGBDiff_loader = torch.utils.data.DataLoader(
    #     TSNDataSet('/home/qx/project/data/UCF101/tvl1_flow/', args.val_list, num_segments=args.num_segments,
    #                new_length=5,
    #                modality='Flow',
    #                image_tmpl="{}/{}/frame{:06d}.jpg",
    #                random_shift=False,
    #                test_mode=True,
    #                On_Video=False,
    #                interval=2,
    #                transform=torchvision.transforms.Compose([
    #                    GroupScale(int(scale_size)),
    #                    GroupCenterCrop(crop_size),
    #                    Stack(roll=args.arch == 'BNInception'),
    #                    ToTorchFormatTensor(div=args.arch != 'BNInception'),
    #                    GroupNormalize(input_mean, input_std)
    #                ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    RGBmodel.eval()
    RGBDiffmodel.eval()
    epoch_prec1 = 0
    epoch_prec5 = 0

    class_num = np.array([0] * num_class)
    class_prec1 = np.array([0] * num_class)
    class_prec5 = np.array([0] * num_class)

    with torch.no_grad():
        for (RGBdata, target), (RGBDiffdata, _) in zip(RGB_loader, RGBDiff_loader):
            #print(RGBdata.shape,RGBDiffdata.shape)
            target = target.to(device)
            RGBDiffdata = RGBDiffdata.to(device)
            RGBdata = RGBdata.to(device)
            RGBoutput = RGBmodel(RGBdata)
            RGBDiffoutput = RGBDiffmodel(RGBDiffdata)
            #print(RGBoutput.shape,RGBDiffoutput.shape)
            output = RGBoutput+RGBDiffoutput

            # _,pred = output.topk(5, 1, True, True)
            # pred=pred.cpu().numpy()
            # target=target.cpu().numpy()
            # for i in range(args.batch_size):
            #     pred_name=index2name(pred[i],'./raw/classInd.txt')
            #     true_name=index2name(target[i:i+1],'./raw/classInd.txt')
            #     print('pred name:',pred_name,'true name:',true_name)


            (prec1, prec5), (class_prec1_t, class_prec5_t), class_num_t \
                = class_accuracy(output.data, target, num_class, topk=(1, 5))

            class_num += class_num_t
            class_prec1 += class_prec1_t
            class_prec5 += class_prec5_t

            #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            epoch_prec1 += prec1.item() * target.size(0)
            epoch_prec5 += prec5.item() * target.size(0)


        epoch_prec1 = 1.0 * epoch_prec1 / len(RGBDiff_loader.dataset)
        epoch_prec5 = 1.0 * epoch_prec5 / len(RGBDiff_loader.dataset)

    print("Accuracy top1: {} top5:{}".format(epoch_prec1, epoch_prec5))

    sorted_name=index2name(np.argsort(class_prec1 / class_num),'./raw/classInd.txt')
    sorted_score=np.sort(class_prec1 / class_num)
    for i in range(num_class):
        print(sorted_name[i],sorted_score[i])

    print()

    sorted_name = index2name(np.argsort(class_prec5 / class_num), './raw/classInd.txt')
    sorted_score = np.sort(class_prec5 / class_num)
    for i in range(num_class):
        print(sorted_name[i], sorted_score[i])





if __name__ == '__main__':
    main()
