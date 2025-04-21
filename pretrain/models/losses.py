import torch
import torch.nn as nn
import torch.nn.functional as F
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