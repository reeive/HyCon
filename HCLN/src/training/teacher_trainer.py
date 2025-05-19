import torch
import torch.nn.functional as F

def train_teacher(net_t, X_t, G, lbls, train_mask, optimizer_t):
    net_t.train()
    optimizer_t.zero_grad()
    outs = net_t(X_t, G)
    loss = F.nll_loss(F.log_softmax(outs[train_mask], dim=1), lbls[train_mask])
    loss.backward()
    optimizer_t.step()
    return loss.item()

@torch.no_grad()
def valid_teacher(net_t, X_t, G, lbls, mask, evaluator):
    net_t.eval()
    outs = net_t(X_t, G)
    res = evaluator.validate(lbls[mask], outs[mask])
    return res

@torch.no_grad()
def test_teacher(net_t, X_t, G, lbls, mask, evaluator, ft_noise_level=0):
    net_t.eval()
    X_t_eval = X_t.clone()
    if ft_noise_level > 0:
        X_t_eval = (1 - ft_noise_level) * X_t_eval + ft_noise_level * torch.randn_like(X_t_eval)
    outs = net_t(X_t_eval, G)
    res = evaluator.test(lbls[mask], outs[mask])
    return res