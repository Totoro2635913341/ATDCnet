import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from torch.utils.data import DataLoader
from net.ATDCnet import Model
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
import random
from tqdm.autonotebook import tqdm, trange
import sys
import torchvision
from combined_loss import *
from metrics_calculation import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch UIE')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--resize', type=int, default=256,help="resize images, default:resize images to 256*256")
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='250', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_train', type=str, default='UIEB/input')
parser.add_argument('--data_train_RMT', type=str, default='UIEB/input_RMT')
parser.add_argument('--label_train', type=str, default='UIEB/label')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--data_test', type=str, default='UIEB/test')
parser.add_argument('--data_test_RMT', type=str, default='UIEB/test_RMT')
parser.add_argument('--label_test', type=str, default='UIEB/test_label')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--save_folder', default='models/train/', help='Location to save checkpoint models')
parser.add_argument('--output_folder', default='results/train/', help='Location to save images')
parser.add_argument('--use_pretrain', type=bool, default=False)
parser.add_argument('--checkpoint', type=str,default="snapshots/.....")
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--vgg_para', type=float, default=0.01)
parser.add_argument('--MSE_para', type=float, default=1)
parser.add_argument('--ssim_para', type=float, default=100)
parser.add_argument('--lab_para', type=float, default=0)
parser.add_argument('--lch_para', type=float, default=0)
parser.add_argument('--L1_para', type=float, default=0)
parser.add_argument('--grad_acc', type=int, default=1)


opt = parser.parse_args()


def seed_torch(seed=opt.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch()
# accelerate training speed
cudnn.benchmark = True


@torch.no_grad()
def test(opt, testDataloader, testModel):
    testModel.eval()
    for i, (image, test_RMT, _, name) in enumerate(testDataloader):
        with torch.no_grad():
            image = image.to(opt.device)
            test_RMT = test_RMT.to(opt.device)
            generate_img = testModel(image, test_RMT)
            torchvision.utils.save_image(generate_img, opt.output_folder + name[0])
    SSIM_BGR, PSNR_BGR, SSIM_YCBCR, PSNR_YCBCR, SSIM_GRAY, PSNR_GRAY,MSE = calculate_metrics_ssim_psnr_all(opt.output_folder, opt.label_test)
    # UIQM_measures, UICM_measures, UISM_measures, UICONM_measures = calculate_UIQM(opt.output_folder)
    return SSIM_BGR, PSNR_BGR, SSIM_YCBCR, PSNR_YCBCR, SSIM_GRAY, PSNR_GRAY, MSE


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if torch.cuda.is_available():
    opt.device = "cuda"
else:
    opt.device = "cpu"



print('===> Loading datasets')

train_set = get_training_set(opt.data_train, opt.data_train_RMT,opt.label_train, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

test_set = get_eval_set(opt.data_test, opt.data_test_RMT, opt.label_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)


print('===> Building model ')

model = Model().cuda()
loss = Multicombinedloss(opt).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

# lr decrease
scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

modelName = "ATDCnet"

device = opt.device

if opt.use_pretrain:
    # Load pretrained models
    model = torch.load(opt.checkpoint)
    print('successfully loading moedl {}！'.format(opt.checkpoint))
else:
    print('No pretrain model found, training will start from scratch！')

last_psnr = 0.700000
last_SSIM = 0.600000

for epoch in trange(opt.start_iter, opt.nEpochs + 1):

    mse_loss_tmp = 0
    vgg_loss_tmp = 0
    total_loss_tmp = 0
    ssim_loss_tmp = 0

    for input, input_RMT, label, _ in tqdm(training_data_loader, desc=f"[Train]", leave=False):

        input = input.to(device)
        input_RMT = input_RMT.to(device)
        label = label.to(device)


        model.train()
        output = model(input, input_RMT)
        output = output.to(device)

        totalLoss, mse_loss, vgg_loss, ssim_loss = loss(output, label)

        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

        mse_loss_tmp += mse_loss.item()
        vgg_loss_tmp += vgg_loss.item()
        ssim_loss_tmp += ssim_loss.item()
        total_loss_tmp += totalLoss.item()


    SSIM_BGR, PSNR_BGR, SSIM_YCBCR, PSNR_YCBCR, SSIM_GRAY, PSNR_GRAY, MSE = test(opt, testing_data_loader, model)

    oldprint = sys.stdout
    # ######################################## log #################################################################
    log_path_suffix = 'test_logs/' + modelName
    file_path = log_path_suffix + '.log'
    sys.stdout = open(file_path, 'a', encoding='utf-8')

    sys.stdout.write(
        "\r[Epoch %d/%d] , [SSIM %f] , [PSNR: %f], [SSIM_YCBCR: %f] , [PSNR_YCBCR: %f], [SSIM_GRAY: %f], [PSNR_GRAY: %f], [MSE: %f]  "
        % (
            epoch,
            opt.nEpochs,
            float(SSIM_BGR),
            float(PSNR_BGR),
            float(SSIM_YCBCR),
            float(PSNR_YCBCR),
            float(SSIM_GRAY),
            float(PSNR_GRAY),
            float(MSE),
        )
    )

    sys.stdout = oldprint

    print("[Epoch %d/%d] , [SSIM %f] , [PSNR: %f] , "
          "[SSIM_YCBCR: %f] , [PSNR_YCBCR: %f], [SSIM_GRAY: %f], [PSNR_GRAY: %f], [MSE: %f] "
          % (epoch, opt.nEpochs, float(SSIM_BGR), float(PSNR_BGR),
             float(SSIM_YCBCR), float(PSNR_YCBCR), float(SSIM_GRAY), float(PSNR_GRAY), float(MSE)))

    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)

    if (PSNR_BGR > last_psnr) and (SSIM_BGR > last_SSIM):
        torch.save(model, opt.save_folder + modelName + '_epoch{}_psnr{}_ssim{}'.format(epoch, PSNR_BGR, SSIM_BGR) + '_best.pth')
        last_psnr = PSNR_BGR
        last_SSIM = SSIM_BGR
    elif (PSNR_BGR > last_psnr) and (SSIM_BGR < last_SSIM):
        torch.save(model, opt.save_folder + modelName + '_epoch{}_psnr{}_ssim{}'.format(epoch, PSNR_BGR, SSIM_BGR) + '_PSNR_best.pth')
        last_psnr = PSNR_BGR

    elif (PSNR_BGR < last_psnr) and (SSIM_BGR > last_SSIM):
        torch.save(model, opt.save_folder + modelName + '_epoch{}_psnr{}_ssim{}'.format(epoch, PSNR_BGR, SSIM_BGR) + '_SSIM_best.pth')
        last_SSIM = SSIM_BGR

    scheduler.step()




