import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractor import build_backbone
from .modules import TransformerDecoder, Transformer
from einops import rearrange


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        x = x_permuted.permute(0, 3, 2, 1)

        return x

class Classifier(nn.Module):
    def __init__(self, in_chan=128, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan, in_chan//2, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan//2),
                            nn.ReLU(),
                            nn.Conv2d(in_chan//2, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x

class token_encoder(nn.Module):
    def __init__(self, in_chan = 32, token_len = 6, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        # self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.contiguous().view([b, c, -1])

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)

        # tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, size = 32, heads = 8):
        super(token_decoder, self).__init__()
        # self.pos_embedding_decoder = nn.Parameter(torch.randn(1, in_chan, size, size))
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape
        # x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class CDNet(nn.Module):
    def __init__(self, backbone='resnet18', upscale = 8, img_size=512, output_stride=16, f_c=64, in_c=3):
        super(CDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.upscale = upscale

        self.extractor1 = build_backbone(backbone, output_stride, BatchNorm, in_c, f_c)
        self.extractor2 = build_backbone(backbone, output_stride, BatchNorm, in_c, f_c)

        self.token_encoder1 = token_encoder(in_chan = f_c)
        self.token_decoder1 = token_decoder(in_chan=f_c, size=img_size // 2)

        self.token_encoder2 = token_encoder(in_chan=f_c)
        self.token_decoder2 = token_decoder(in_chan = f_c, size = img_size//2)

        # self.OSU = DUpsampling(f_c, upscale+1)
        self.classifier = Classifier(f_c*2, 2)

    def forward(self, lr_img, hr_img):

        x1 = self.extractor1(lr_img)
        x2 = self.extractor2(hr_img)

        x2_lr = F.interpolate(x2, size=x1.shape[2:], mode='bicubic', align_corners=True)
        x2_inter = F.interpolate(x2_lr, size=x2.shape[2:], mode='bicubic', align_corners=True)

        x1_inter = F.interpolate(x1, size=x2.shape[2:], mode='bicubic', align_corners=True)
        ti1 = self.token_encoder1(x1_inter)
        ti2 = self.token_encoder1(x2_inter)
        tl2 = self.token_encoder1(x2)
        x1_inter = self.token_decoder1(x1_inter, ti1)

        t1 = self.token_encoder2(x1_inter)
        t2 = self.token_encoder2(x2)
        x1 = self.token_decoder2(x1_inter, t1)
        x2 = self.token_decoder2(x2, t2)

        x = torch.cat([x1,x2], dim=1)
        x = F.interpolate(x, size=hr_img.shape[2:], mode='bicubic', align_corners=True)
        x = self.classifier(x)

        return x, ti2, tl2


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()