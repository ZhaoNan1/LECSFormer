import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self,img_size=224, patch_size=4, in_channels=3, embed_dim=64, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] 

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.activate1 = nn.GELU()
        self.activate2 = nn.GELU()
        self.proj1 = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=2, padding=1)
        self.proj2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

            
    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W} doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.activate1(x)
        x = self.proj2(x)
        x = self.norm2(x)
        x = self.activate2(x)
        x = x.flatten(2).transpose(1,2)
        return x
        


class BasicLayer(nn.Module):
    
    def __init__(self, dim, input_resolution, depth, num_heads, split_size,
                mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                drop_path=0.1, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if depth == 1:
            self.blocks = nn.ModuleList([
                LECSWinBlock(dim=dim, num_heads=num_heads, resolution=input_resolution, mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size, drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[0] if isinstance(drop_path,list) else drop_path, norm_layer=norm_layer)
            ])
        else:
            self.blocks = nn.ModuleList([
                LECSWinBlock(dim=dim, num_heads=num_heads, resolution=input_resolution, mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size, drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[0] if isinstance(drop_path,list) else drop_path, norm_layer=norm_layer),

                SLECSWinBlock(dim=dim, num_heads=num_heads, resolution=input_resolution, mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size, drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[1] if isinstance(drop_path,list) else drop_path, norm_layer=norm_layer)
            ])
        # patch merge layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,dim//2, dim)
        else:
            self.downsample = None
    
    def forward(self,x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk,x)
            else:
                x = blk(x)
        return x

        

class LECSWinBlock(nn.Module):
    def __init__(self, dim, resolution, num_heads,
                split_size=7, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                drop=0.1, attn_drop=0.1, drop_path=0.1,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 2
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            CSWMHSA(
                dim // 2, resolution=self.resolution, idx=i,
                split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                qk_scale=qk_scale,attn_drop=attn_drop)
            for i in range(self.branch_num)
            ])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.lem = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.nwc = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU())
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = LEMLP(resolution=resolution,in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
        self.norm2 = norm_layer(dim)
        
    def forward(self,x):
        H, W = self.resolution[0], self.resolution[1]
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        short_cut = x
        ## LEM
        x_ = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        lem = self.lem(x_)
        lem = rearrange(lem, ' b c h w -> b (h w) c', h=H, w=W)
        ## attn
        qkv = self.qkv(x).reshape(B, H, W, 3, C).permute(0, 3, 4, 1, 2)
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :C//2, :, :])
            x2 = self.attns[1](qkv[:, :, C//2:, :, :])
            attended_x = torch.cat([x1,x2],dim=2)
        else:
            attended_x = self.attns[0](qkv)
        attended_x = self.norm1(self.proj(attended_x))  # post-norm
        x = short_cut + self.drop_path(attended_x) + lem

        # nwc
        nwc = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        nwc = self.nwc(nwc)
        nwc = rearrange(nwc, ' b c h w -> b (h w) c', h=H, w=W)
        x = x + nwc

        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class SLECSWinBlock(nn.Module):
    def __init__(self, dim, resolution, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0.1, attn_drop=0.1, drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = resolution
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            SCSWMHSA(
                dim // 2, resolution=self.patches_resolution, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop)
            for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.lem = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU())
        self.nwc = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU())
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = LEMLP(resolution=resolution, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                        drop=drop,
                        )
        self.norm2 = norm_layer(dim)
    def forward(self, x):
        H, W = self.patches_resolution[0], self.patches_resolution[1]
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        short_cut = x
        ## LEM
        x_ = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        lem = self.lem(x_)
        lem = rearrange(lem, ' b c h w -> b (h w) c', h=H, w=W)
        ## attn
        qkv = self.qkv(x).reshape(B, H, W, 3, C).permute(0, 3, 4, 1, 2)
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :C // 2, :, :])
            x2 = self.attns[1](qkv[:, :, C // 2:, :, :])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.norm1(self.proj(attened_x))
        x = short_cut + self.drop_path(attened_x) + lem
        # nwc
        nwc = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        nwc = self.nwc(nwc)
        nwc = rearrange(nwc, ' b c h w -> b (h w) c', h=H, w=W)
        x = x + nwc
        x = x + self.drop_path(self.norm2(self.mlp(x))) 
        return x


class LEMLP(nn.Module):
    def __init__(self, resolution, in_features=None, hidden_features=None, act_layer=nn.GELU, drop=0.1, ):
        super().__init__()
        self.resolution = resolution
        self.linear1 = nn.Sequential(nn.Linear(in_features, hidden_features),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, groups=hidden_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_features),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_features, in_features))
        self.dim = in_features
        self.hidden_dim = hidden_features
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        hh, ww = self.resolution[0], self.resolution[1]
        x = self.linear1(x)
        x = self.drop(x)
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)
        x = self.dwconv(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=ww)
        x = self.linear2(x)
        x = self.drop(x)
        return x



class Merge_Block(nn.Module):
    def __init__(self, resolution, dim, dim_out, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.resolution = resolution
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)
        self.act = nn.GELU()

    def forward(self, x):
        B, new_HW, C = x.shape
        H, W = self.resolution[0]*2, self.resolution[1]*2
        assert new_HW == H * W, 'error size'
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.act(self.norm(self.conv(x)))
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        return x


def img2windows(img, H_sp, W_sp):    
    B,C,H,W = img.shape
    img = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img = img.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1,H_sp*W_sp,C)
    return img

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

def img2windows_shuffle(img, H_sp, W_sp, head):
    img = rearrange(img, 'b (h d) (H_sp hh) (W_sp ww) -> (b hh ww) h (H_sp W_sp) d', h=head, H_sp=H_sp, W_sp=W_sp)
    return img

def windows2img_shuffle(img_splits_hw, H_sp, W_sp, H, W, head):
    B = int(img_splits_hw.shape[0] / (H * W // H_sp // W_sp))
    img = rearrange(img_splits_hw, '(b hh ww) h (H_sp W_sp) d -> b (h d) (H_sp hh) (W_sp ww)', b=B, hh=H // H_sp,
                    h=head, H_sp=H_sp, W_sp=W_sp)
    img = img.contiguous().view(B, H, W, -1)
    return img


class CSWMHSA(nn.Module):
    def __init__(self,dim, resolution, idx, split_size, num_heads, dim_out=None, attn_drop=0.1,qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution[0], self.resolution[1]
        elif idx == 0:
            H_sp, W_sp = self.resolution[0], self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution[1], self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)
    
    def img2cswin(self, x):
        _,C,_,_ = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)   
        x = x.reshape(-1, self.H_sp*self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self,x,func):
        _,C,_,_ = x.shape
        v = self.img2cswin(x) # value
        x = self.img2cswin(x).reshape(-1, C, self.H_sp, self.W_sp)  # B', C, H', W'
        lepe = func(x).reshape(-1,self.num_heads, C // self.num_heads, self.H_sp*self.W_sp).permute(0,1,3,2).contiguous()
        return v, lepe
        
    def forward(self, x):
        B, _, C, H, W = x.shape
        ## pading for split windows
        H_pad = (self.H_sp - H % self.H_sp) % self.H_sp
        W_pad = (self.W_sp - W % self.W_sp) % self.W_sp
        top_pad = H_pad // 2
        down_pad = H_pad - top_pad
        left_pad = W_pad // 2
        right_pad = W_pad - left_pad
        H_ = H + H_pad
        W_ = W + W_pad

        qkv = F.pad(x, [left_pad, right_pad, top_pad, down_pad])  ### B,3,C,H',W'
        qkv = qkv.permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # img2windows
        q = self.img2cswin(q) * self.scale
        k = self.img2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        attn = (q @ k.transpose(-2,-1))
        attn = nn.functional.softmax(attn,dim=-1,dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v) + lepe
        x = x.transpose(1,2).reshape(-1, self.H_sp*self.W_sp, C)

        # windows2img
        x = windows2img(x, self.H_sp, self.W_sp, H_, W_)
        x = x[:, top_pad:H + top_pad, left_pad:W + left_pad, :]
        x = x.reshape(B, -1, C)
        return x


class SCSWMHSA(nn.Module):
    def __init__(self, dim, resolution, idx, split_size, num_heads, dim_out=None, attn_drop=0.1,qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution[0], self.resolution[1]
        elif idx == 0:
            H_sp, W_sp = self.resolution[0], self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution[1], self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin_shuffle(self, x):
        x = img2windows_shuffle(x, self.H_sp, self.W_sp, self.num_heads)
        return x

    def get_lepe(self, x, func):
        B, C, H, W = x.shape
        v = self.im2cswin_shuffle(x) # value
        x = x.view(B, C, self.H_sp, H // self.H_sp, self.W_sp, W // self.W_sp)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous().reshape(-1, C, self.H_sp, self.W_sp)  
        lepe = func(x) 
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return v, lepe

    def forward(self, temp):
        B, _, C, H, W = temp.shape
        ### padding for split window
        H_pad = (self.H_sp - H % self.H_sp) % self.H_sp
        W_pad = (self.W_sp - W % self.W_sp) % self.W_sp
        top_pad = H_pad // 2
        down_pad = H_pad - top_pad
        left_pad = W_pad // 2
        right_pad = W_pad - left_pad
        H_ = H + H_pad
        W_ = W + W_pad

        qkv = F.pad(temp, [left_pad, right_pad, top_pad, down_pad])  ### B,3,C,H',W'
        qkv = qkv.permute(1, 0, 2, 3, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        ### Img2Window
        q = self.im2cswin_shuffle(q)
        k = self.im2cswin_shuffle(k)
        v, lepe = self.get_lepe(v, self.get_v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v) + lepe

        ### Window2Img
        x = windows2img_shuffle(x, self.H_sp, self.W_sp, H_, W_, self.num_heads)  # B H_ W_ C
        x = x[:, top_pad:H + top_pad, left_pad:W + left_pad, :]
        x = x.reshape(B, -1, C)
        return x


class UP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.act2 = nn.GELU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x


class CONVM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.act2 = nn.GELU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = nn.Conv2d(in_channels, out_channels, 3 ,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_last = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_last2 = nn.Conv2d(out_channels, num_class, kernel_size=1, bias=False)

    def forward(self, up_feature):
        x = torch.cat(up_feature, dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        feature = self.act1(self.bn1(feature))  

        x = self.avgpool(x)
        x = self.act2(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv_last(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv_last2(x)
        return x


class Fuse(nn.Module):

    def __init__(self,in_,out,scale,num_class):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(out, num_class, 1, bias=False)

    def forward(self, up_inp):
        outputs = F.interpolate(up_inp, scale_factor=self.scale, mode='bilinear')
        outputs = self.conv1(outputs)
        outputs = self.activation(outputs)
        outputs = self.conv2(outputs)
        return outputs


class LECSFormer(nn.Module):
    '''
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_channels (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 64
        depths (tuple(int)): Depth of decoder.
        num_heads (tuple(int)): Number of attention heads.
        split_size (int): split size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    '''
    def __init__(self,
                img_size=224, patch_size=4, in_channels=3, num_classes=1, embed_dim=64,
                depths=[2, 2, 2, 2],
                num_heads= [2, 4, 8, 16],
                split_size=[1, 3, 7, 7],
                mlp_ratio=4, qkv_bias=False, qk_scale=None,
                drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, patch_norm=True,
                use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_class = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim)
        self.patches_resolution = self.patch_embed.patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()

        # Encoder
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               split_size=split_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=Merge_Block if (
                                       i_layer > 0) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # Decoder
        self.decode1_1 = UP(embed_dim * 8, embed_dim * 4)
        self.decode1_2 = UP(embed_dim * 4, embed_dim * 2)
        self.decode1_3 = UP(embed_dim * 2, embed_dim)

        self.decode2_1 = UP(embed_dim * 4, embed_dim * 2)
        self.decode2_2 = UP(embed_dim * 2, embed_dim)

        self.decode3_1 = UP(embed_dim * 2, embed_dim)

        self.decode4_1 = CONVM(embed_dim, embed_dim)

        self.ffm = FeatureFusionModule(embed_dim * 4, embed_dim, num_class=self.num_class)

        self.aux1 = Fuse(embed_dim, embed_dim, 4, num_class=self.num_class)
        self.aux2 = Fuse(embed_dim, embed_dim, 4, num_class=self.num_class)
        self.aux3 = Fuse(embed_dim, embed_dim, 4, num_class=self.num_class)
        self.aux4 = Fuse(embed_dim, embed_dim, 4, num_class=self.num_class)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        H, W = self.patches_resolution[0],self.patches_resolution[1]
        x = self.patch_embed(x)
        stage = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            temp = rearrange(x, ' b (h w) (c) -> b c h w ', h=H // (2 ** i), w=W // (2 ** i))
            stage.append(temp)
        return stage 

    def forward_up_features(self, stage):
        x_1_1 = self.decode1_1(stage[3])
        x_1_2 = self.decode1_2(x_1_1)
        x_1_3 = self.decode1_3(x_1_2)

        x_2_1 = self.decode2_1(torch.add(stage[2],x_1_1))
        x_2_2 = self.decode2_2(torch.add(x_2_1,x_1_2))

        x_3_1 = self.decode3_1(stage[1] + x_2_1 + x_1_2)

        x_4_1 = self.decode4_1(stage[0] + x_1_3 + x_2_2 + x_3_1)

        return [x_1_3, x_2_2, x_3_1, x_4_1]

    def forward(self, x):
        stage = self.forward_features(x)
        up_feature = self.forward_up_features(stage)
        x = self.ffm(up_feature)
        aux1 = self.aux1(up_feature[0])
        aux2 = self.aux2(up_feature[1])
        aux3 = self.aux3(up_feature[2])
        aux4 = self.aux4(up_feature[3])
        return x, [aux1,aux2,aux3,aux4]





















