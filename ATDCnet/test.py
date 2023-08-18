import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch.utils.data import DataLoader
import argparse
from data import get_eval_set
import random
import torchvision
import torch
from metrics_calculation import *

# Testing settings
parser = argparse.ArgumentParser(description='PyTorch UIE')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--resize', type=int, default=256,help="resize images, default:resize images to 256*256")
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--data_test', type=str, default='UIEB/test')
parser.add_argument('--data_test_RMT', type=str, default='UIEB/test_RMT')
parser.add_argument('--label_test', type=str, default='UIEB/test_label')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--output_folder', default='results/test/', help='Location to save images')
parser.add_argument('--use_pretrain', type=bool, default=True)
parser.add_argument('--checkpoint', type=str,default="checkpoint/snapshots/UIEB.pth")



opt = parser.parse_args()


def seed_torch(seed=opt.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch()


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if torch.cuda.is_available():
    opt.device = "cuda"
else:
    opt.device = "cpu"





print('===> Loading testdataset')

test_set = get_eval_set(opt.data_test, opt.data_test_RMT, opt.label_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)


print('===> Loading model ')


if opt.use_pretrain:
    # Load pretrained models
    model = torch.load(opt.checkpoint)
    print('successfully loading moedl {} ！'.format(opt.checkpoint))
else:
    print('No pretrain model found, training will start from scratch！')


for i, (image, test_RMT, _, name) in enumerate(testing_data_loader):
    image = image.to(opt.device)
    test_RMT = test_RMT.to(opt.device)
    generate_img = model(image, test_RMT)
    torchvision.utils.save_image(generate_img, opt.output_folder + name[0])

SSIM_BGR, PSNR_BGR, SSIM_YCBCR, PSNR_YCBCR, SSIM_GRAY, PSNR_GRAY,MSE = calculate_metrics_ssim_psnr_all(opt.output_folder, opt.label_test)

print("[SSIM %f] , [PSNR: %f] , [SSIM_YCBCR: %f] , [PSNR_YCBCR: %f], [SSIM_GRAY: %f], [PSNR_GRAY: %f], [MSE: %f] "
      % (float(SSIM_BGR), float(PSNR_BGR), float(SSIM_YCBCR), float(PSNR_YCBCR), float(SSIM_GRAY), float(PSNR_GRAY), float(MSE)))






