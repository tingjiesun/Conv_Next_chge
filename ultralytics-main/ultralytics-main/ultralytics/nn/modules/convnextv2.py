import torch
import torch.nn as nn
import torch.nn.functional as F

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # 简化的 trunc_normal 实现，避免依赖 timm
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block. """
    def __init__(self, c1, c2=None, k=None, s=None, p=None, drop_path=0.):
        # c1: input channels, c2: output channels. In ConvNeXt blocks, c1 should equal c2.
        super().__init__()
        if c2 is None: c2 = c1
        # assert c1 == c2, f"ConvNeXtV2 Block expects same input/output channels, got {c1} and {c2}"
        dim = c1
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2Stem(nn.Module):
    """ Stem layer: 4x downsampling """
    def __init__(self, c1, c2, k=4, s=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s),
            LayerNorm(c2, eps=1e-6, data_format="channels_first")
        )
    
    def forward(self, x):
        return self.stem(x)

class ConvNeXtV2Downsample(nn.Module):
    """ Downsampling layer: 2x downsampling """
    def __init__(self, c1, c2, k=2, s=2):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm(c1, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(c1, c2, kernel_size=k, stride=s),
        )
    
    def forward(self, x):
        return self.downsample(x)
