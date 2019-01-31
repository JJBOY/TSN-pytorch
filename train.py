from torch.nn.utils import clip_grad_norm
from utils import *
import time
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(path,train_loader, model, criterion, optimizer, epoch, clip_gradient=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.to(device)
        data = data.to(device)
        output = model(data)
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        optimizer.zero_grad()
        loss.backward()

        if clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), clip_gradient)
            if total_norm > clip_gradient:
                print("clipping gradient{} with coef {}". \
                      format(total_norm, clip_gradient / total_norm))
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    info = {'Epoch': [epoch],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg, 5)],
            'Prec@1': [round(top1.avg, 4)],
            'Prec@5': [round(top5.avg, 4)],
            'lr': optimizer.param_groups[0]['lr']
            }
    record_info(info, path+'train.csv', 'train')


def validate(path,val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            target = target.to(device)
            data = data.to(device)
            output = model(data)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    info = {'Epoch': [epoch],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg, 5)],
            'Prec@1': [round(top1.avg, 4)],
            'Prec@5': [round(top5.avg, 4)],
            }
    if path is not None:
        record_info(info, path+'test.csv', 'test')
    else:
        print(info)
    return top1.avg

def test(val_loader, model, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()

    class_num = np.array([0] * num_classes)
    class_prec1 = np.array([0] * num_classes)
    class_prec5 = np.array([0] * num_classes)

    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            target = target.to(device)
            data = data.to(device)
            output = model(data)

            #def class_accuracy(outputs, targets, num_classes, topk=(1,), ):
            (prec1, prec5) ,( class_prec1_t ,class_prec5_t ), class_num_t  \
                = class_accuracy(output.data, target,num_classes ,topk=(1, 5))

            class_num+=class_num_t
            class_prec1+=class_prec1_t
            class_prec5+=class_prec5_t

            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    info = {
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Prec@1': [round(top1.avg, 4)],
            'Prec@5': [round(top5.avg, 4)],
            }

    print(info)


    print(np.argsort(class_prec1/class_num))
    print(np.sort(class_prec1/class_num))
    print()
    print(np.argsort(class_prec5/class_num))
    print(np.sort(class_prec5/class_num))

