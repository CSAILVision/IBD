import time
import os
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from easydict import EasyDict as edict
import settings
import numpy as np

EXP_SETTINGS = edict({
    "print_freq": 10,
    "batch_size": 64,
    "data_imagenet": "/home/sunyiyou/dataset/imagenet/",
    "data_places365": "/home/sunyiyou/dataset/places365_standard/",
    "workers": 12,
})

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

def test_clf_power(feat_clf, model):
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        model.cuda()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % EXP_SETTINGS.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg

    # weight_label = list(model.parameters())[-2].data.numpy()
    # norm_weight_label = np.linalg.norm(weight_label, axis=1)
    # rankings, errvar, coefficients, residuals_T = np.load(os.path.join(settings.OUTPUT_FOLDER, "decompose_pos.npy"))
    # new_weight = (weight_label / norm_weight_label[:, None] - residuals_T.T) * norm_weight_label[:, None]
    # new_weight = residuals_T.T * norm_weight_label[:, None]
    # model.fc.weight.data.copy_(torch.Tensor(new_weight))
    criterion = nn.CrossEntropyLoss().cuda()
    valdir = os.path.join(EXP_SETTINGS.data_places365, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=EXP_SETTINGS.batch_size, shuffle=False,
        num_workers=EXP_SETTINGS.workers, pin_memory=True)

    prec1 = validate(val_loader, model, criterion)
    print(prec1)
