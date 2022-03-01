import math
import numpy as np
from os.path import join as pjoin
from collections import OrderedDict
import copy
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from deform_conv import DeformConv2d
from SoftPool import soft_pool2d, SoftPool2d
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    # StdConv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3))
    def forward(self, x):
        w = self.weight  # [64, 3, 7, 7] 64 ge channel 3 7x7
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        # x = x.transpose([0, 3, 1, 2])
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        # self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]  # 12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768/12
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    #PreActBottleneck(cin=width, cout=width*4, cmid=width) 64 256 64
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin  # 256(3) 512(4) 1024(9)
        cmid = cmid or cout // 4  # 64(3)  128(4) 256(9)

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = [patch_size[0] // 2, patch_size[1] // 2]
        patches_resolution = [img_size[0] // stride[0], img_size[1] // stride[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=2)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    #print(x.shape)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        shortcut = x  # ([12, 4096, 96])
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        # jia conv

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows fenqu chuangkou
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dcn = DeformConv2d(dim, 2 * dim, 3, padding=1, stride=2, modulation=True)
        self.dconv = nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, dilation=2, padding=2)
        self.conv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1)
        # in_channels,out_channels,kernel_size,stride = stride,padding = padding,bias = not (use_batchnorm),
        self.conv2 = nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=2)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(2 * dim)
        self.relu = nn.ReLU(inplace=True)
        self.gn1 = nn.GroupNorm(dim // 2, dim)
        self.gn2 = nn.GroupNorm(dim, 2 * dim)
        self.pool = SoftPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.gelu = nn.GELU()

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape  # ([12, 4096, 96])

        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)  # ([12, 64, 64, 384])
        x = x.view(B, C, H, W)
        # offset = Variable(torch.randn(N, num_deformable_groups * 2 * kH * kW, inH, inW).cuda(), requires_grad=True)
        # x = self.relu(self.bn2(self.dcn(x)))
        short = x
        x = self.gelu(self.bn2(self.conv1(x)))

        x = self.bn2(self.dconv(x))

        x = self.gelu(self.bn2(self.conv3(x)))
        short = self.gn1(self.pool(short))

        short = self.gelu(self.bn2(self.conv2(short)))

        x = x + short
        x = x.view(B, -1, 2 * C)  # B H/2*W/2 4*C

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, no_down=0, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.no_down = no_down
        self.dcn3 = DeformConv2d(dim, dim // 2, 3, padding=1, stride=1, modulation=True)
        self.dcn5 = DeformConv2d(dim, dim // 2, 5, padding=1, stride=1, modulation=True)
        self.dcn7 = DeformConv2d(dim, dim // 2, 7, padding=1, stride=1, modulation=True)
        self.bn = nn.BatchNorm2d(dim // 2)
        self.relu = nn.ReLU(inplace=True)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv11 = conv1x1(dim, dim // 2, bias=False)
        self.gelu = nn.GELU()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2 = nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, padding=0)
        self.dcn1 = DeformConv2d(dim, dim // 2, 3, padding=1, stride=1, modulation=True)
        self.dcn2 = DeformConv2d(dim // 2, dim, 3, padding=1, stride=1, modulation=True)
        self.bn1 = nn.BatchNorm2d(dim // 2)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        # Trans_feature
        i = 0
        for blk in self.blocks:
            if i % 2 == 0:

                s = x
                H, W = self.input_resolution
                B, L, C = s.shape  # ([12, 4096, 96])

                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                s = s.view(B, H, W, C)  # ([12, 64, 64, 384])
                s = s.view(B, C, H, W)

                s = self.gelu(self.bn1(self.conv1(s)))
                ss = s
                s_h, s_w = self.pool_h(s), self.pool_w(s)  # .permute(0, 1, 3, 2)
                s = torch.matmul(s_h, s_w)

                s = self.gelu(self.bn2(self.conv2(s)))
            x = blk(x)
            i=i+1
            if i % 2 == 0:
                H, W = self.input_resolution
                B, L, C = x.shape  # ([12, 4096, 96])
                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                x = x.view(B, H, W, C)  # ([12, 64, 64, 384])
                x = x.view(B, C, H, W)
                x=x+s
                x = x.view(B, -1, C)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class depthwise_separable_conv(nn.Module):
    def init(self, in_channels, out_channels):
        super(depthwise_separable_conv, self).init()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU()  # dilation=2,
        # print(in_channels, out_channels)//2
        self.relu2 = nn.ReLU()
        self.dcn = DeformConv2d(in_channels, out_channels, 3, padding=0, stride=1, modulation=True)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv5 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # print(x.shape)

        x = self.relu1(self.bn1(self.conv4(x)))
        # print(x.shape)
        short = x
        # x=self.relu2(self.conv4(x))

        x_h, x_w = self.pool_h(x), self.pool_w(x)  # .permute(0, 1, 3, 2)
        x = torch.matmul(x_h, x_w)
        x = x + short
        out = torch.sigmoid(self.bn2(self.conv5(x)))
        # out = torch.mul(short, out)
        # print(short.shape, out.shape)
        return short, out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAtt(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max','soft']):
        super(ChannelAtt, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU()
            #nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        self.incr = nn.Linear(gate_channels // reduction_ratio, gate_channels)

    def forward(self, x):
        channel_att_sum = None
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avgpoolmlp = self.mlp(avg_pool)
        maxpoolmlp=self.mlp(max_pool)
        pooladd = avgpoolmlp+maxpoolmlp

        self.pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        soft_pool = self.mlp(self.pool(x))
        weightPool = soft_pool * pooladd
        # channel_att_sum = self.mlp(weightPool)
        channel_att_sum = self.incr(weightPool)
        Att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return Att



class TransResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.#self.hybrid_model = Res
    img_size=224, patch_size=4, in_chans=3, embed_dim=96,NetV2((3,4,9),1)"""

    # block_units 3,4,9
    def __init__(self, config, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width  # 64
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)]

            ))),  # i=2,3
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),  # i=2,3,4
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),  # i=2,3,4,5,6,7,8,9
            ('block4', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 16, cout=width * 32, cmid=width * 8, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 32, cout=width * 32, cmid=width * 8)) for i in
                 range(2, block_units[3] + 1)],
            ))),
        ]))
        self.num_classes = config.n_classes  # 6
        depths = config.trans.depths
        self.num_layers = len(depths)
        self.patch_norm = True
        embed_dim = config.trans.embed_dim  # 96
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.embeddings = PatchEmbed(img_size=config.data.img_size, patch_size=8, in_chans=config.data.in_chans,
                                     embed_dim=embed_dim,
                                     norm_layer=nn.LayerNorm if self.patch_norm else None)
        # num_patches = self.embeddings.num_patches
        patches_resolution = self.embeddings.patches_resolution
        self.patches_resolution = patches_resolution

        self.mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint = False
        # stochastic depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        '''    config.trans.num_heads = [3, 6, 12, 24]
    config.trans.depths = [2, 2, 6, 2] '''
        # w=32
        outchannel = [width * 4, width * 8, width * 16, width * 32]
        self.layers = nn.ModuleList()
        self.Down_features = nn.ModuleList()
        self.Conv3 = nn.ModuleList()
        self.Fuse = nn.ModuleList()
        self.Dcn = nn.ModuleList()
        # self.Bn = nn.ModuleList()
        for i_layer in range(self.num_layers):  # i_layer 0,1,2,3
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=config.trans.num_heads[i_layer],
                               window_size=config.trans.window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               no_down=i_layer,
                               # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            downsample = PatchMerging(input_resolution=layer.input_resolution, dim=layer.dim, norm_layer=norm_layer)
            # Conv3x3 = BasicConv(in_planes=layer.dim, out_planes=outchannel[i_layer], kernel_size=3, stride=1, padding=2,
            #                   dilation=2, groups=1, relu=True, bn=True, bias=False)
            Conv3x3 = BasicConv(in_planes=layer.dim, out_planes=outchannel[i_layer], kernel_size=1, stride=1,
                                groups=1, relu=True, bn=True, bias=False)
            dcn = DeformConv2d(inc=outchannel[i_layer], outc=outchannel[i_layer], kernel_size=3, padding=0, stride=1,
                               modulation=True)
            # bn=nn.BatchNorm2d(outchannel[i_layer])
            fuse = ChannelAtt(gate_channels=outchannel[i_layer], reduction_ratio=2, pool_types=['avg', 'max'])
            # CoordAttention(in_channels=layer.dim, out_channels=outchannel[i_layer])
            self.Dcn.append(dcn)
            # self.Bn.append(bn)
            self.Fuse.append(fuse)
            self.Conv3.append(Conv3x3)
            self.Down_features.append(downsample)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # self.encoder = Encoder(config) #transformer

    def forward(self, x):
        Trans_features = []
        b, c, in_size, _ = x.shape  # torch.Size([12, 3, 256, 256])
        #print('www',x.shape)
        trans_x = x
        trans_x = self.embeddings(trans_x)  # [12, 4096, 96]
        i_layer = 0
        self.num_layers = len(self.layers)
        for layer in self.layers:
            trans_x = layer(trans_x)
            B, n_patch, hidden = trans_x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            Trans_x = trans_x.permute(0, 2, 1)
            Trans_x = Trans_x.contiguous().view(B, hidden, h, w)
            Trans_features.append(Trans_x)

            if (i_layer < self.num_layers - 1):
                trans_x = self.Down_features[i_layer](trans_x)
            i_layer = i_layer + 1

        # resnet50
        features = []
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)

        for i in range(len(self.body)):
            x = self.body[i](x)
            x = self.Dcn[i](x)
            s1=self.Conv3[i](Trans_features[i])
            short = self.Fuse[i](s1)
            short = short * x

            x = x + short+s1
            if i < len(self.body) - 1:
                features.append(x)
        return x, features[::-1]


