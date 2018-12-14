import os
import pandas as pd


def accuracy(outputs, targets, topk=(1,)):
    # compute the topk accuracy

    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)  # return the topk scores in every input
    pred = pred.t()  # shape:(maxk,N)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def record_info(info, filename, mode):
    if mode == 'train':
        result = (
            'Batch Time {batch_time} '
            'Epoch Time {epoch_time} '
            'Data {data_time} \n'
            'Loss {loss} '
            'Prec@1 {top1} '
            'Prec@5 {top5}\n'
            'LR {lr}\n'.format(batch_time=info['Batch Time'], epoch_time=info['Epoch Time'],
                               data_time=info['Data Time'], loss=info['Loss'],
                               top1=info['Prec@1'], top5=info['Prec@5'],lr=info['lr']))
        print(result)

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch', 'Batch Time', 'Data Time', 'Loss', 'Prec@1', 'Prec@5', 'lr']

    if mode == 'test':
        result = (
            'Batch Time {batch_time} '
            'Epoch Time {epoch_time} \n'
            'Loss {loss} '
            'Prec@1 {top1} '
            'Prec@5 {top5} \n'.format(batch_time=info['Batch Time'], epoch_time=info['Epoch Time'],
                                      loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))
        print(result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch', 'Batch Time', 'Epoch Time', 'Loss', 'Prec@1', 'Prec@5']

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else:  # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False, columns=column_names)

def adjust_learning_rate(optimizer):
    optimizer.lr=optimizer.lr*0.2
    for param_group in optimizer.param_groups:
        param_group['lr'] = optimizer.lr * param_group['lr_mult']
