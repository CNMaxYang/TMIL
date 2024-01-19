from timm.models import vision_transformer
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MIL(nn.Module):
    def __init__(self):
        super(MIL, self).__init__()
        self.vit = vision_transformer.vit_small_patch16_224_in21k(num_classes=5, pretrained=True)
        self.norm1 = nn.LayerNorm(384)
        self.norm2 = nn.LayerNorm(128)
        self.att1 = Attention(384, 6)
        self.att2 = Attention(128, 4)
        self.MLP1 = nn.Linear(384, 128)
        self.MLP2 = nn.Linear(128, 1)
        self.cls = nn.Linear(16, 5)
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, 384))
        # trunc_normal_(self.pos_embed, std=.02)

    def patchify(self, imgs):
        input_patch = []
        for i in range(4):
            for j in range(4):
                input_patch.append(imgs[:, :, i*224:(i+1)*224, j*224:(j+1)*224])
        return input_patch

    def forward(self, x):
        input_patch = self.patchify(x)
        y = self.vit.forward_features(input_patch[0])
        y = torch.unsqueeze(y, 1)
        for i in range(1, 16):
            y = torch.cat((y, torch.unsqueeze(self.vit.forward_features(input_patch[i]), 1)), dim=1)
        y = y + self.pos_embed
        y = self.norm1(y)
        y = self.att1(y)
        y = self.MLP1(y)
        y = self.norm2(y)
        y = self.att2(y)
        y = self.MLP2(y)
        y = torch.squeeze(y)
        y = self.cls(y)
        return y
