import torch, torch.nn as nn, torch.nn.functional as F
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch.fft as fft
import cv2

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # b,256,256
        # b, 8,256,32
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                # FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PatchEmbedding(nn.Module): # 이건 conv로 하는거
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 256, img_size: int = 256):
        self.patch_size = patch_size
        super().__init__()
        patch_height=patch_size
        patch_width=patch_size
        # self.projection = nn.Sequential(
        #     # using a conv layer instead of a linear one -> performance gains
        #     nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
        #     Rearrange('b e (h) (w) -> b (h w) e'),
        # )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(((patch_height * patch_width) * in_channels), emb_size),
        )

        ## rfft flatten
        self.pos_embedding = nn.Parameter(torch.randn(1,img_size//2,emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape

        x = self.to_patch_embedding(x) # b,256,129*1
        x += self.pos_embedding
        return x


# START


class preconv_seven(nn.Module):  ################################################################################################ SEVEN
    def __init__(self, enet_type=None, out_dim=4, img_size=256, patch_size=16, pre_conv_kernel=7, n_classes=1000, dim=256, depth=12, heads=12, mlp_dim=256, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(preconv_seven,self).__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = PatchEmbedding(in_channels=channels,patch_size=patch_size,emb_size=dim,img_size=img_size)

        self.dropout = nn.Dropout(emb_dropout) # transformer 들어가기 전

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.avgpoold2d = nn.AdaptiveAvgPool2d((16,16))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 256
            nn.Linear(dim, out_dim),
        )

        self.preconv1 = nn.Sequential(
            nn.Conv2d(16, channels, kernel_size=pre_conv_kernel, stride=1, padding=(pre_conv_kernel//2)),
            nn.GELU()
        )

        self.const_weight = nn.Parameter(torch.randn(size=[16, 1, 5, 5]), requires_grad=True)
    def normalized_F(self):
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        for i in range(16):
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, 2, 2] = -1.0

    def forward(self, img):
        ## pre-processing layer
        self.normalized_F()
        img = F.conv2d(img, self.const_weight, padding=2)
        img = self.preconv1(img)

        ## Frequency positional encoding layer
        ## rfft2 is added one column.
        img = torch.abs(fft.rfft2(img, dim=(2, 3), norm='ortho'))
        ## we removed first column.
        img = img[:,:,:,1:]
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        ## transformer encoder
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim=1)# if self.pool == 'mean' else x[:, 0]

        ## regression layerg
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x
