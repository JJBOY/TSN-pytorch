from torch.nn.utils import clip_grad_norm
from utils import *
import time
import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(train_loader,model,criterion,optimizer,epoch,clip_gradient=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    end=time.time()
    for i,(data,target) in enumerate(train_loader):
        data_time.update(time.time()-end)
        target=target.to(device)
        data=data.to(device)
        output=model(data)
        loss=criterion(output,target)
        prec1,prec5=accuracy(output.data,target,topk=(1,5))
        losses.update(loss.item(),data.size(0))
        top1.update(prec1.item(),data.size(0))
        top5.update(prec5.item(),data.size(0))

        optimizer.zero_grad()
        loss.backward()

        if clip_gradient is not None:
            total_norm=clip_grad_norm(model.parameters(),clip_gradient)
            if total_norm>clip_gradient:
                print("clipping gradient{} with coef {}".\
                    format(total_norm,clip_gradient/total_norm))
        optimizer.step()

        batch_time.update(time.time() - end)
        end=time.time()

    info = {'Epoch': [epoch],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg, 5)],
            'Prec@1': [round(top1.avg, 4)],
            'Prec@5': [round(top5.avg, 4)],
            'lr': optimizer.param_groups[0]['lr']
            }
    record_info(info, 'record/train.csv', 'train')


def validate(val_loader,model,criterion,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end=time.time()

    with torch.no_grad():
        for i,(data,target) in enumerate(val_loader):
            data_time.update(time.time()-end)
            target=target.to(device)
            data=data.to(device)
            output=model(data)
            loss=criterion(output,target)
            prec1,prec5=accuracy(output.data,target,topk=(1,5))
            losses.update(loss.item(),data.size(0))
            top1.update(prec1.item(),data.size(0))
            top5.update(prec5.item(),data.size(0))
            batch_time.update(time.time() - end)
            end=time.time()

    info = {'Epoch': [epoch],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg, 5)],
            'Prec@1': [round(top1.avg, 4)],
            'Prec@5': [round(top5.avg, 4)],
            }
    record_info(info, 'record/test.csv', 'test')
    return top1.avg