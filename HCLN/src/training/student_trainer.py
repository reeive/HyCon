import torch
import torch.nn as nn
import torch.nn.functional as F

class HighOrderConstraint(nn.Module):
    def __init__(self, teacher_model_hc, X_teacher_hc, G_hc, noise_level=1.0, tau=1.0):
        super().__init__()
        teacher_model_hc.eval()
        self.tau = tau
        pred = teacher_model_hc(X_teacher_hc, G_hc).softmax(dim=-1).detach()
        entropy_x = -(pred * pred.log()).sum(1, keepdim=True)
        entropy_x[entropy_x.isnan()] = 0
        entropy_e = G_hc.v2e(entropy_x, aggr="mean")

        X_noise = X_teacher_hc.clone() * (torch.randn_like(X_teacher_hc) + 1) * noise_level
        pred_ = teacher_model_hc(X_noise, G_hc).softmax(dim=-1).detach()
        entropy_x_ = -(pred_ * pred_.log()).sum(1, keepdim=True)
        entropy_x_[entropy_x_.isnan()] = 0
        entropy_e_ = G_hc.v2e(entropy_x_, aggr="mean")

        self.delta_e_ = (entropy_e_ - entropy_e).abs()
        if self.delta_e_.max() > 0:
             self.delta_e_ = 1 - self.delta_e_ / self.delta_e_.max()
        else:
            self.delta_e_ = torch.ones_like(self.delta_e_)
        self.delta_e_ = self.delta_e_.squeeze()

    def forward(self, pred_s, pred_t, G_hc):
        pred_s_softmax = F.softmax(pred_s, dim=1)
        pred_t_softmax = F.softmax(pred_t, dim=1)

        e_mask_probs = self.delta_e_
        e_mask_probs = torch.clamp(e_mask_probs, 0.0, 1.0)
        e_mask = torch.bernoulli(e_mask_probs).bool()
        
        if not e_mask.any():
            return torch.tensor(0.0, device=pred_s.device)

        pred_s_e = G_hc.v2e(pred_s_softmax, aggr="mean")[e_mask]
        pred_t_e = G_hc.v2e(pred_t_softmax, aggr="mean")[e_mask]
        
        if pred_s_e.numel() == 0:
            return torch.tensor(0.0, device=pred_s.device)
        
        loss = F.kl_div(torch.log(pred_s_e / self.tau + 1e-9), pred_t_e / self.tau, reduction="batchmean", log_target=False)
        return loss

def train_stu(net_s, X_s, G, lbls_s, out_t, train_mask_s, optimizer_s, hc=None, lamb=0, tau_kd=1.0):
    net_s.train()
    optimizer_s.zero_grad()
    
    outs_s = net_s(X_s)
    
    loss_k_node = F.kl_div(
        F.log_softmax(outs_s / tau_kd, dim=1),
        F.softmax(out_t / tau_kd, dim=1),
        reduction="batchmean",
        log_target=False
    )
    
    current_loss_k = loss_k_node
    
    if hc is not None:
        loss_h = hc(outs_s, out_t, G)
        current_loss_k = current_loss_k + loss_h

    loss_x = torch.tensor(0.0, device=outs_s.device)
    if lamb > 0 and lbls_s is not None and train_mask_s is not None and train_mask_s.sum() > 0:
        loss_x = F.nll_loss(F.log_softmax(outs_s[train_mask_s], dim=1), lbls_s[train_mask_s])

    loss = loss_x * lamb + current_loss_k * (1 - lamb)
    
    loss.backward()
    optimizer_s.step()
    return loss.item()

@torch.no_grad()
def valid_stu(net_s, X_s, lbls_s, mask_s, evaluator_s, num_segmentation_classes):
    net_s.eval()
    outs_s = net_s(X_s)
    if lbls_s is not None and mask_s is not None and evaluator_s is not None and mask_s.sum() > 0 :
        res = evaluator_s.validate(lbls_s[mask_s], outs_s[mask_s])
        return res
    return 0.0

@torch.no_grad()
def test_stu(net_s, X_s, lbls_s, mask_s, evaluator_s, num_segmentation_classes, ft_noise_level=0):
    net_s.eval()
    X_s_eval = X_s.clone()
    if ft_noise_level > 0:
        X_s_eval = (1 - ft_noise_level) * X_s_eval + ft_noise_level * torch.randn_like(X_s_eval)
    
    logits_s = net_s(X_s_eval)
    soft_pseudo_labels = F.softmax(logits_s, dim=1)

    test_metrics = None
    if lbls_s is not None and mask_s is not None and evaluator_s is not None and mask_s.sum() > 0:
        test_metrics = evaluator_s.test(lbls_s[mask_s], logits_s[mask_s])

    return soft_pseudo_labels, test_metrics