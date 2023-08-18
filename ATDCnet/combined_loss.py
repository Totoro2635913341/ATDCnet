from VGG_loss import *
from torchvision import models
import pytorch_ssim
import torch

class Multicombinedloss(nn.Module):
    def __init__(self, config):

        super(Multicombinedloss, self).__init__()
        # load vgg19
        vgg19 = models.vgg19_bn(pretrained=False)
        vgg19.load_state_dict(torch.load('vgg19_bn-c79401a0.pth'))

        self.p_vgg=config.vgg_para
        self.p_MSE=config.MSE_para
        self.p_ssim=config.ssim_para

        self.vggloss = VGG_loss(vgg19, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config.device)
        self.l1loss = nn.L1Loss().to(config.device)
        self.ssim = pytorch_ssim.SSIM().to(config.device)


    def forward(self, out, label):

        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.p_MSE*self.mseloss(out, label)
        vgg_loss = self.p_vgg*self.l1loss(inp_vgg, label_vgg)
        ssim_loss= -(self.p_ssim*self.ssim(out,label))

        total_loss = mse_loss + vgg_loss + ssim_loss
        return total_loss, mse_loss, vgg_loss ,ssim_loss

