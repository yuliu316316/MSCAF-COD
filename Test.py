import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lib.MSCSFNet import Network
from Src.utils.Dataloader import test_dataset1
from Src.utils.trainer import eval_mae, numpy2tensor
import imageio
from skimage import img_as_ubyte
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='/media/liuyu/PycharmProjects/MSCSF-Net/pytorch_lib/snapshot/MSCSF-Net/LNet_100.pth')
parser.add_argument('--test_save', type=str,
                    default='./evaluation/MSCSF-Net-CHAMELEON/')
opt = parser.parse_args()

model = Network().cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['CHAMELEON']:
    save_path = opt.test_save
    os.makedirs(save_path, exist_ok=True)
    test_loader = test_dataset1(image_root='./Dataset/TestDataset/{}/Imgs/'.format(dataset),
                               gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                               edge_root='./Dataset/TestDataset/{}/Edge/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        cam, _1, _2, _3, _4 = model(image)
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        imageio.imsave(save_path+name, img_as_ubyte(cam))
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

print("\n[Congratulations! Testing Done]")
