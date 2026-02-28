import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Supporting functions and modules (adapted from conv_layers.py for clarity)
def get_act(nonlinearity='elu'):
    if nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)

def conv1x1(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    """1x1 convolution"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv

def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, kernel_size=3):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=dilation, bias=bias, dilation=dilation)

class CRPBlock(nn.Module):
    """Chained Residual Pooling block for feature refinement."""
    def __init__(self, features, n_stages=2, act=nn.ELU(inplace=True), maxpool=True):
        super().__init__()
        self.convs = nn.ModuleList([conv3x3(features, features, bias=False) for _ in range(n_stages)])
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2) if maxpool else nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act
        self.n_stages = n_stages

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.convs[i](path)
            x = x + path
        return x

class RCUBlock(nn.Module):
    """Residual Convolutional Unit block for adaptation."""
    def __init__(self, features, n_blocks=2, n_stages=2, act=nn.ELU(inplace=True)):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, f'{i+1}_{j+1}_conv', conv3x3(features, features, bias=False))

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, f'{i+1}_{j+1}_conv')(x)
            x = x + residual
        return x

class MSFBlock(nn.Module):
    """Multi-Scale Fusion block for combining features."""
    def __init__(self, in_planes, features):
        super().__init__()
        self.convs = nn.ModuleList([conv3x3(ip, features) for ip in in_planes])

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.convs[0].out_channels, *shape, device=xs[0].device)
        for i, conv in enumerate(self.convs):
            h = conv(xs[i])
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums = sums + h
        return sums

class RefineBlock(nn.Module):
    """RefineNet block for upsampling and fusion."""
    def __init__(self, in_planes, features, act=nn.ELU(inplace=True), start=False, end=False):
        super().__init__()
        self.n_blocks = len(in_planes)
        self.adapt_convs = nn.ModuleList([RCUBlock(ip, act=act) for ip in in_planes])
        self.output_convs = RCUBlock(features, n_blocks=3 if end else 1, act=act)
        if not start:
            self.msf = MSFBlock(in_planes, features)
        self.crp = CRPBlock(features, act=act)

    def forward(self, xs, output_shape):
        hs = [self.adapt_convs[i](xs[i]) for i in range(self.n_blocks)]
        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]
        h = self.crp(h)
        h = self.output_convs(h)
        return h

class ResidualBlock(nn.Module):
    """Residual block with optional downsampling and dilation."""
    def __init__(self, in_channels, out_channels, resample=None, act=nn.ELU(inplace=True),
                 normalization=nn.InstanceNorm2d, dilation=1, adjust_padding=False):
        super().__init__()
        self.act = act
        self.resample = resample
        self.norm1 = normalization(in_channels)
        self.norm2 = normalization(out_channels)
        if resample == 'down':
            self.shortcut = ConvMeanPool(in_channels, out_channels, adjust_padding=adjust_padding)
            self.conv1 = conv3x3(in_channels, out_channels, dilation=dilation)
            self.conv2 = ConvMeanPool(out_channels, out_channels, adjust_padding=adjust_padding)
        else:
            self.shortcut = conv1x1(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
            self.conv1 = conv3x3(in_channels, out_channels, dilation=dilation)
            self.conv2 = conv3x3(out_channels, out_channels, dilation=dilation)

    def forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class ConvMeanPool(nn.Module):
    """Mean pooling followed by conv for downsampling."""
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            self.conv = conv3x3(input_dim, output_dim, kernel_size=kernel_size, bias=biases)
        else:
            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv3x3(input_dim, output_dim, kernel_size=kernel_size, bias=biases))

    def forward(self, inputs):
        output = self.conv(inputs)
        return (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4.0

# Time embedding function (from convs.py)
def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = np.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb

# Main ConvNet adapted for baselines
class ConvNet(nn.Module):
    """Adapted ConvNet for baselines: predicts raw mean direction (d dims)."""
    def __init__(self, in_channels=2, ngf=128, nonlinearity='elu', normalization=nn.InstanceNorm2d):
        super().__init__()
        self.in_channels = in_channels  # Ambient dim (e.g., 2 for S1)
        self.ngf = ngf
        self.act = get_act(nonlinearity)
        self.norm = normalization

        # Initial conv and time projection
        self.begin_conv = conv3x3(in_channels, ngf)
        self.fc_t1 = nn.Linear(ngf, ngf)

        # Residual stages with downsampling and dilations
        self.res1 = nn.ModuleList([
            ResidualBlock(ngf, ngf, resample=None, act=self.act, normalization=self.norm),
            ResidualBlock(ngf, ngf, resample=None, act=self.act, normalization=self.norm)
        ])
        self.fc_t2 = nn.Linear(ngf, ngf)

        self.res2 = nn.ModuleList([
            ResidualBlock(ngf, 2 * ngf, resample='down', act=self.act, normalization=self.norm),
            ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=self.act, normalization=self.norm)
        ])
        self.fc_t3 = nn.Linear(ngf, 2 * ngf)

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * ngf, 2 * ngf, resample='down', act=self.act, normalization=self.norm, dilation=2),
            ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=self.act, normalization=self.norm, dilation=2)
        ])
        self.fc_t4 = nn.Linear(ngf, 2 * ngf)

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * ngf, 2 * ngf, resample='down', act=self.act, normalization=self.norm, dilation=4, adjust_padding=True),
            ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=self.act, normalization=self.norm, dilation=4)
        ])
        self.fc_t5 = nn.Linear(ngf, 2 * ngf)

        # Refinement decoder
        self.refine1 = RefineBlock([2 * ngf], 2 * ngf, act=self.act, start=True)
        self.refine2 = RefineBlock([2 * ngf, 2 * ngf], 2 * ngf, act=self.act)
        self.refine3 = RefineBlock([2 * ngf, 2 * ngf], ngf, act=self.act)
        self.refine4 = RefineBlock([ngf, ngf], ngf, act=self.act, end=True)

        # Final layers: output raw_mu (d channels) + raw_kappa (1 channel)
        self.final_norm = self.norm(ngf)
        self.final_conv = conv3x3(ngf, in_channels)

    def forward(self, x, t):
        # Input: x [B, d, H, W], t [B]
        # Scale input to [-1,1] range for stability
        output = self.begin_conv(2 * x - 1)

        B = x.shape[0]

        # ---- make t into shape [B] ----
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        t = t.to(device=x.device, dtype=x.dtype)

        if t.ndim == 0:
            # scalar -> [B]
            t = t.expand(B)
        else:
            # e.g. [B,1,1,1] / [B,1] -> [B]
            t = t.reshape(-1)
            if t.numel() == 1:
                t = t.expand(B)
            elif t.numel() != B:
                raise ValueError(f"t has {t.numel()} elements, expected {B}")

        # Add time embedding
        t_embed = get_time_embedding(t, self.ngf)
        output = output + self.fc_t1(t_embed)[:, :, None, None]

        # Encoder path with residuals and time additions
        layer1 = output
        for res in self.res1:
            layer1 = res(layer1)
        layer1 = layer1 + self.fc_t2(t_embed)[:, :, None, None]

        layer2 = layer1
        for res in self.res2:
            layer2 = res(layer2)
        layer2 = layer2 + self.fc_t3(t_embed)[:, :, None, None]

        layer3 = layer2
        for res in self.res3:
            layer3 = res(layer3)
        layer3 = layer3 + self.fc_t4(t_embed)[:, :, None, None]

        layer4 = layer3
        for res in self.res4:
            layer4 = res(layer4)
        layer4 = layer4 + self.fc_t5(t_embed)[:, :, None, None]

        # Decoder path with refinement
        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])        
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        # Final processing: norm, act, conv to params
        output = self.final_norm(output)
        output = self.act(output)
        # assert torch.isfinite(output).all(), "NaN in forward before final_conv"
        output = self.final_conv(output)  # [B, d, H, W] 
        # assert torch.isfinite(output).all(), "NaN after final_conv"
        return output

class ConvNet_DVFM(nn.Module):
    """Adapted ConvNet for DVFM: predicts raw mean direction (d dims) and raw kappa (1 dim)."""
    def __init__(self, in_channels=2, ngf=128, nonlinearity='elu', normalization=nn.InstanceNorm2d):
        super().__init__()
        self.in_channels = in_channels  # Ambient dim (e.g., 2 for S1)
        self.ngf = ngf
        self.act = get_act(nonlinearity)
        self.norm = normalization

        # Initial conv and time projection
        self.begin_conv = conv3x3(in_channels, ngf)
        self.fc_t1 = nn.Linear(ngf, ngf)

        # Residual stages with downsampling and dilations
        self.res1 = nn.ModuleList([
            ResidualBlock(ngf, ngf, resample=None, act=self.act, normalization=self.norm),
            ResidualBlock(ngf, ngf, resample=None, act=self.act, normalization=self.norm)
        ])
        self.fc_t2 = nn.Linear(ngf, ngf)

        self.res2 = nn.ModuleList([
            ResidualBlock(ngf, 2 * ngf, resample='down', act=self.act, normalization=self.norm),
            ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=self.act, normalization=self.norm)
        ])
        self.fc_t3 = nn.Linear(ngf, 2 * ngf)

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * ngf, 2 * ngf, resample='down', act=self.act, normalization=self.norm, dilation=2),
            ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=self.act, normalization=self.norm, dilation=2)
        ])
        self.fc_t4 = nn.Linear(ngf, 2 * ngf)

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * ngf, 2 * ngf, resample='down', act=self.act, normalization=self.norm, dilation=4, adjust_padding=True),
            ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=self.act, normalization=self.norm, dilation=4)
        ])
        self.fc_t5 = nn.Linear(ngf, 2 * ngf)

        # Refinement decoder
        self.refine1 = RefineBlock([2 * ngf], 2 * ngf, act=self.act, start=True)
        self.refine2 = RefineBlock([2 * ngf, 2 * ngf], 2 * ngf, act=self.act)
        self.refine3 = RefineBlock([2 * ngf, 2 * ngf], ngf, act=self.act)
        self.refine4 = RefineBlock([ngf, ngf], ngf, act=self.act, end=True)

        # Final layers: output raw_mu (d channels) + raw_kappa (1 channel)
        self.final_norm = self.norm(ngf)
        # self.final_conv = conv3x3(ngf, in_channels)
        self.mu_head = conv3x3(ngf, in_channels)
        self.kappa_head = conv3x3(ngf, 1)

    def forward(self, x, t):
        # Input: x [B, d, H, W], t [B]
        # Scale input to [-1,1] range for stability
        output = self.begin_conv(2 * x - 1)

        B = x.shape[0]

        # ---- make t into shape [B] ----
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        t = t.to(device=x.device, dtype=x.dtype)

        if t.ndim == 0:
            # scalar -> [B]
            t = t.expand(B)
        else:
            # e.g. [B,1,1,1] / [B,1] -> [B]
            t = t.reshape(-1)
            if t.numel() == 1:
                t = t.expand(B)
            elif t.numel() != B:
                raise ValueError(f"t has {t.numel()} elements, expected {B}")

        # Add time embedding
        t_embed = get_time_embedding(t, self.ngf)
        output = output + self.fc_t1(t_embed)[:, :, None, None]

        # Encoder path with residuals and time additions
        layer1 = output
        for res in self.res1:
            layer1 = res(layer1)
        layer1 = layer1 + self.fc_t2(t_embed)[:, :, None, None]

        layer2 = layer1
        for res in self.res2:
            layer2 = res(layer2)
        layer2 = layer2 + self.fc_t3(t_embed)[:, :, None, None]

        layer3 = layer2
        for res in self.res3:
            layer3 = res(layer3)
        layer3 = layer3 + self.fc_t4(t_embed)[:, :, None, None]

        layer4 = layer3
        for res in self.res4:
            layer4 = res(layer4)
        layer4 = layer4 + self.fc_t5(t_embed)[:, :, None, None]

        # Decoder path with refinement
        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])        
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        # Final processing: norm, act, conv to params
        output = self.final_norm(output)
        output = self.act(output)
        # assert torch.isfinite(output).all(), "NaN in forward before final_conv"
        # output_mu = self.mu_head(output)  # [B, d, H, W] 
        mu_raw = self.mu_head(output)  # [B, d, H, W]
        output_mu = mu_raw / mu_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
        output_mu = output_mu.clamp(min=1e-6)
        output_mu = output_mu / output_mu.norm(dim=1, keepdim=True).clamp(min=1e-8)
        
        kappa_raw = self.kappa_head(output)  # [B, 1, H, W] 
        output_kappa = F.softplus(kappa_raw) + 1e-4       # >0, 避免 0
        output_kappa = output_kappa.clamp(max=200.0)
        # assert torch.isfinite(output).all(), "NaN after final_conv"
        return output_mu, output_kappa

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act == 'relu' else nn.ELU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SimpleCNN(nn.Module):
    """
    For SFM / Fisher-flow:
    - forward(x, t) -> v_ambient  with shape [B, 2, H, W]
    - expects to use time as an extra input channel (so in_channels = 2 + 1 in evaluate_binary.py)
    """
    def __init__(self, in_channels=3, hidden_channels=(32, 64, 128), out_channels=2, act='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        hs = list(hidden_channels)
        self.b1 = ConvBlock(in_channels, hs[0], act=act)
        self.b2 = ConvBlock(hs[0], hs[1], act=act)
        self.b3 = ConvBlock(hs[1], hs[2], act=act)

        self.down1 = nn.AvgPool2d(2)  # 28 -> 14
        self.down2 = nn.AvgPool2d(2)  # 14 -> 7

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 7 -> 14
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14 -> 28

        self.head = nn.Conv2d(hs[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        x: [B, 2, H, W]
        t: [B] or scalar
        """
        B, C, H, W = x.shape

        # Make t channel: [B,1,H,W]
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(B)
        t_ch = t.view(B, 1, 1, 1).to(dtype=x.dtype).expand(B, 1, H, W)

        # If model expects time as channel, concat it.
        # evaluate_binary.py instantiates in_channels=2+1, so this path will be used.
        if C + 1 == self.in_channels:
            inp = torch.cat([x, t_ch], dim=1)
        elif C == self.in_channels:
            inp = x
        else:
            raise ValueError(f"Channel mismatch: x has {C} ch, model expects {self.in_channels}")

        # Encoder
        h1 = self.b1(inp)          # [B, hs0, 28, 28]
        h2 = self.b2(self.down1(h1))  # [B, hs1, 14, 14]
        h3 = self.b3(self.down2(h2))  # [B, hs2, 7, 7]

        # Decoder (lightweight)
        u2 = self.up1(h3)          # [B, hs2, 14, 14]
        u2 = u2[:, :h2.shape[1]]   # simple channel align (safe if hs2>=hs1; if not, remove this line)
        u2 = u2 + h2               # skip

        u1 = self.up2(u2)          # [B, hs1, 28, 28]
        u1 = u1[:, :h1.shape[1]]
        u1 = u1 + h1               # skip

        v = self.head(u1)          # [B, 2, 28, 28]
        return v


class SimpleCNN_dvfm(nn.Module):
    """
    For DVFM sampling in evaluate_binary.py:
    - forward(x, t) -> (mu, kappa)
      mu   : [B, d, H, W] on unit sphere
      kappa: [B, 1, H, W] positive
    """
    def __init__(self, embed_dim=2, hidden_channels=(32, 64, 128), act='relu'):
        super().__init__()
        self.d = int(embed_dim)
        hs = list(hidden_channels)

        # input: x has d channels, we concat time => d+1
        self.b1 = ConvBlock(self.d + 1, hs[0], act=act)
        self.b2 = ConvBlock(hs[0], hs[1], act=act)
        self.b3 = ConvBlock(hs[1], hs[2], act=act)

        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # two heads
        self.mu_head = nn.Conv2d(hs[0], self.d, kernel_size=1)
        self.kappa_head = nn.Conv2d(hs[0], 1, kernel_size=1)

    def forward(self, x, t):
        """
        x: [B, d, H, W]  (points on sphere)
        t: [B] or scalar
        """
        B, C, H, W = x.shape
        if C != self.d:
            raise ValueError(f"SimpleCNN_dvfm expects x with {self.d} channels, got {C}")

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(B)
        t_ch = t.view(B, 1, 1, 1).to(dtype=x.dtype).expand(B, 1, H, W)

        inp = torch.cat([x, t_ch], dim=1)  # [B, d+1, H, W]

        # Encoder
        h1 = self.b1(inp)                # [B, hs0, 28, 28]
        h2 = self.b2(self.down1(h1))     # [B, hs1, 14, 14]
        h3 = self.b3(self.down2(h2))     # [B, hs2, 7, 7]

        # Decoder
        u2 = self.up1(h3)
        u2 = u2[:, :h2.shape[1]]
        u2 = u2 + h2

        u1 = self.up2(u2)
        u1 = u1[:, :h1.shape[1]]
        u1 = u1 + h1

        mu_raw = self.mu_head(u1)        # [B, d, H, W]
        mu = mu_raw / mu_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)

        kappa_raw = self.kappa_head(u1)  # [B, 1, H, W]
        kappa = F.softplus(kappa_raw) + 1e-4

        return mu, kappa