import torch
import torchvision

from torch.autograd import Variable
from datetime import datetime
import os
from apex import amp
from torch import nn
import pytorch_ssim
import pytorch_iou
def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    """
    recalibrate the misdirection in the training
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.contiguous().view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(c*d)

# bce_loss = nn.BCELoss(size_average=True)
# ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
# iou_loss = pytorch_iou.IOU(size_average=True)
#
# def bce_ssim_loss(lateral_edge1, edges):
#
# 	bce_out = bce_loss(lateral_edge1, edges)
# 	ssim_out = 1 - ssim_loss(lateral_edge1, edges)
# 	iou_out = iou_loss(lateral_edge1, edges)
#
# 	loss = bce_out + ssim_out + iou_out
#
# 	return loss

def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :return:
    """
    # model.train()
    # size_rates = [0.75, 1, 1.25]
    # for step, data_pack in enumerate(train_loader):
    #     for rate in size_rates:
    #         optimizer.zero_grad()
    #         images, gts, edges = data_pack
    #         images = Variable(images).cuda()
    #         gts = Variable(gts).cuda()
    #         edges = Variable(edges).cuda()
    #         if rate != 1:
    model.train()
    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts, edges = data_pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()

        map_24, map_23, map_22, map_21, out21, out22, out23, out24, lateral_edge1, lateral_edge = model(images)
        loss1 = loss_func(map_24, gts)
        loss2 = loss_func(map_23, gts)
        loss3 = loss_func(map_22, gts)
        loss4 = loss_func(map_21, gts)

        loss5 = loss_func(out21, gts)
        loss6 = loss_func(out22, gts)
        loss7 = loss_func(out23, gts)
        loss8 = loss_func(out24, gts)

        # loss_edge1 = criterior_edge(lateral_edge1, edges)
        # loss_edge = criterior_edge(lateral_edge, edges)
        loss_edge1 = nn.BCEWithLogitsLoss()(lateral_edge1, edges)
        loss_edge = nn.BCEWithLogitsLoss()(lateral_edge, edges)
        loss_out = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8

        loss_total = loss_out + 3 * loss_edge + 4 * loss_edge1

        with amp.scale_loss(loss_total, optimizer) as scale_loss:
            scale_loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()



        if step % 50 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss: {:.4f}, Loss: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_out.data, loss_edge.data))

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'SINet_%d.pth' % (epoch+1))


