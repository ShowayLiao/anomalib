import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(self.norm, torch.nn.SyncBatchNorm), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class BaseFPN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_feature_levels):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        if x.size()[2:] != y.size()[2:]:
            x = F.interpolate(x, size=(h, w), mode='bilinear')
        return x


class FPN(BaseFPN):
    def __init__(self, in_channels, hidden_dim, num_feature_levels):
        super().__init__(in_channels, hidden_dim, num_feature_levels)

        self.bottomup_conv = nn.ModuleList()
        self.conv_proj = nn.ModuleList()

        for _ in range(num_feature_levels - 1):
            self.bottomup_conv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels[_], in_channels[_ + 1], kernel_size=2, stride=2),
                    Conv_LayerNorm(in_channels[_ + 1]),
                    nn.GELU(),
                )
            )

        for _ in range(num_feature_levels):
            self.conv_proj.append(
                nn.Sequential(
                    Conv2d(
                        in_channels[_],
                        hidden_dim[_],
                        kernel_size=1,
                        bias=False,
                        norm=Conv_LayerNorm(hidden_dim[_]),
                    ),
                    Conv2d(
                        hidden_dim[_],
                        hidden_dim[_],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm=Conv_LayerNorm(hidden_dim[_]),
                        groups=hidden_dim[_],
                    ),
                )
            )

    def forward(self, srcs, masks=None):
        feature_maps = srcs[::-1]
        up_results = [feature_maps[0]]
        results = [up_results[0]]

        for feature, upconv in zip(feature_maps[1:], self.bottomup_conv):
            up_feature = self._upsample_add(upconv(up_results[-1]), feature)
            up_results.append(up_feature)
            results.append(up_feature)

        return [self.conv_proj[i](f) for i, f in enumerate(results)]
