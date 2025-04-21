import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np


# from nt_xent import NT_Xent
# from losses import *


class nt_xent(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size=1):
        super(nt_xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

        # self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        self.batch_size = z_i.shape[0]
        self.mask = self.mask_correlated_samples(self.batch_size, 1)

        z_i = torch.flatten(z_i, start_dim=1)
        z_j = torch.flatten(z_j, start_dim=1)
        N = 2 * self.batch_size * self.world_size
        if self.batch_size > z_i.shape[0]:
            return 0

        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            smooth = 1e-6
            return torch.log(torch.sum(F_loss) + smooth)

class BinaryDiceLoss_xent(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss_xent, self).__init__()
    def _dice_loss(self, score, target):
        smooth = 1e-6
        dim=(2,3,4)
        intersect = torch.sum(score * target,dim=dim)
        y_sum = torch.sum(target * target,dim=dim)
        # y_sum2 = torch.sum(target * target.permute(1,0,2,3,4),dim=dim)
        z_sum = torch.sum(score * score,dim=dim)
        # z_sum2 = torch.sum(score * score.permute(1,0,2,3,4),dim=dim)
        dice_loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # dice_loss2 = (2 * intersect + smooth) / (z_sum2 + y_sum2 + smooth)
        return dice_loss

    def forward(self, inputs, target):
        assert inputs.size()[2:] == target.size()[2:], 'predict & target shape do not match'
        target = target.float()
        dice_loss = self._dice_loss(inputs, target)
        # loss = 1 - dice_loss
        return dice_loss

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
    def _dice_loss(self, score, target):
        smooth = 1e-6
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice_loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        target = target.float()
        dice_loss = self._dice_loss(inputs, target)
        loss = 1 - dice_loss
        return loss
class MSELoss_xent(nn.Module):
    def __init__(self):
        super(MSELoss_xent, self).__init__()
    def _mse_loss(self, scource, target):
        #dim=(2,3,4)
        dim = (1)
        # aa=torch.sum((scource-target)**2,dim=dim)
        # print("aa:",aa)
        # mse_loss = aa / scource.shape[1]
        mse_loss = torch.sum((scource-target)**2,dim=dim) / scource.shape[1]
        return mse_loss

    def forward(self, inputs, target):
        assert inputs.size()[2:] == target.size()[2:], 'predict & target shape do not match'
        target = target.float()
        mse_loss = self._mse_loss(inputs, target)
        # loss = 1 - dice_loss
        return mse_loss
def logits_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes logits on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert input_logits.size() == target_logits.size()
    mse_loss = (input_logits-target_logits)**2
    return mse_loss


class DSGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--beta_R', type=float, default=1.0, help="weight for reconstructive loss (X -> Y' -> X'') and (Y -> X' -> Y'')")
            parser.add_argument('--t', type=float, default=0.1, help='tempeture of contrastive loss')
            parser.add_argument('--lambda_C', type=float, default=0.1, help='weight for contrastive loss')
            parser.add_argument('--gamma_T', type=float, default=10, help='weight for translation loss')
            

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class. 

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'R_A', 'T_A', 'D_B', 'G_B', 'R_B','T_B','con']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_B), D_B (D_A)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_m_1d1u', opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionReconstructive = torch.nn.L1Loss()
            self.criterionTranslation = torch.nn.L1Loss()
            self.criterionPair = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionContrastive = nt_xent(opt.batch_size, 0.1, self.device, 1) # opt.batch_size, opt.temperature:1, self.device, world size(equals to node size):1

            self.optimizer_G_A = torch.optim.Adam(itertools.chain(self.netG.module.UNet_mode1.parameters(),self.netG.module.UNet_share.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_B = torch.optim.Adam(itertools.chain(self.netG.module.UNet_mode2.parameters(),self.netG.module.UNet_share.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_B)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.fake_A,  self.content_RAu3, self.content_RAu2, self.content_RAu1, self.content_RA5, self.content_RA4,self.content_RA3, self.content_RBu3, self.content_RBu2, self.content_RBu1, self.content_RB5, self.content_RB4, self.content_RB3 = self.netG(x1=self.real_A, x2=self.real_B) #self.content_RA, self.content_RB

        self.rec_B, self.rec_A,  self.content_FAu3, self.content_FAu2, self.content_FAu1, self.content_FA5, self.content_FA4, self.content_FA3, self.content_FBu3, self.content_FBu2, self.content_FBu1, self.content_FB5, self.content_FB4, self.content_FB3= self.netG(x1=self.fake_A, x2=self.fake_B) # self.content_FA, self.content_FB


    def backward_D_Basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_Basic(self.netD_A, self.real_B, fake_B)
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_Basic(self.netD_B, self.real_A, fake_A)



    def backward_G_A(self):
        """Calculate the loss for generators G_A and G_B"""
        beta_R = self.opt.beta_R 
        gamma_T = self.opt.gamma_T 
        lambda_C = self.opt.lambda_C
        # GAN loss D_A(G_A(A))
        predict_fake_B = self.netD_A(self.fake_B)
        self.loss_G_A = self.criterionGAN(predict_fake_B, True) 
        # Backward reconstructive loss || G_A(G_B(B)) - B||
        self.loss_R_A = self.criterionReconstructive(self.rec_A, self.real_A) * beta_R
        self.loss_T_A = self.criterionPair(self.fake_B, self.real_B) * gamma_T   
        self.loss_con = self.criterionContrastive(self.content_RBu1, self.content_RAu1) * lambda_C
        self.loss_G_A = self.loss_R_A + self.loss_T_A + self.loss_G_A + self.loss_con
        self.loss_G_A.backward()
    def backward_G_B(self):
        """Calculate the loss for generators G_A and G_B"""
        beta_R = self.opt.beta_R 
        gamma_T = self.opt.gamma_T 
        lambda_C = self.opt.lambda_C
        # GAN loss D_B(G_B(B))
        predict_fake_A = self.netD_B(self.fake_A)
        self.loss_G_B = self.criterionGAN(predict_fake_A, True) 
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_R_B = self.criterionReconstructive(self.rec_B, self.real_B) * beta_R
        self.loss_T_B = self.criterionPair(self.fake_A, self.real_A) * gamma_T    
        self.loss_con = self.criterionContrastive(self.content_RBu1, self.content_RAu1) * lambda_C
        self.loss_G_B = self.loss_R_B + self.loss_T_B + self.loss_G_B + self.loss_con#+ self.loss_idt_B
        self.loss_G_B.backward()
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G_A.zero_grad()  # set G_share G_A and G_B's gradients to zero
        self.backward_G_A()             # calculate gradients for G_A and G_B
        self.optimizer_G_A.step()       # update G_A and G_B's weights
        self.forward()      # compute fake images and reconstruction images.
        self.optimizer_G_B.zero_grad()  
        self.backward_G_B()             # calculate gradients for G_A and G_B
        self.optimizer_G_B.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
