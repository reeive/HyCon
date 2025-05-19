import torch
import torch.nn as nn
from dhg.nn import GCNConv as DHG_GCNConv, HGNNConv as DHG_HGNNConv, MLP as DHG_MLP

class MyGCN(nn.Module):
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.layers0 = DHG_GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.layers1 = DHG_GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)

    def forward(self, X: torch.Tensor, g: "dhg.Graph", get_emb=False) -> torch.Tensor:
        emb = self.layers0(X, g)
        X_out = self.layers1(emb, g)
        if get_emb:
            return emb
        else:
            return X_out

class MyHGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers0 = DHG_HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.layers1 = DHG_HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph", get_emb=False) -> torch.Tensor:
        emb = self.layers0(X, hg)
        X_out = self.layers1(emb, hg)
        if get_emb:
            return emb
        else:
            return X_out

class MyMLPs(nn.Module):
    def __init__(self, dim_in, dim_hid, n_classes) -> None:
        super().__init__()
        self.layer0 = DHG_MLP([dim_in, dim_hid])
        self.layer1 = nn.Linear(dim_hid, n_classes)

    def forward(self, X, get_emb=False):
        emb = self.layer0(X)
        X_out = self.layer1(emb)
        if get_emb:
            return emb
        else:
            return X_out