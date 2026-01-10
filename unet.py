import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups, emb_dim):
        """
        Implement:
            conv -> relu -> norm -> + -> conv -> relu -> norm
                                    ^
                                    |
                        linear -> relu

        Args:
            in_channels: number of channels of the input feature map.
            out_channels: number of channels of the output feature map.
            num_groups: number of groups in the GroupNorm.
            emb_dim: number of embedding dimensions of the time encoding.
        """

        super().__init__()

        self.conv1 = nn.Sequential(
            # The padding preserves the feature dimensions.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, out_channels)
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups, out_channels)
        )

    def forward(self, x, t):
        return self.conv2(self.conv1(x) + self.time_mlp(t)[:, :, None, None])


class Down(nn.Module):

    def __init__(self, in_channels, num_groups, emb_dim):
        """
        Implement a downsampling layer and a ConvBlock.

        Args:
            in_channels: number of channels of the input feature map.
            num_groups: number of groups in the GroupNorm.
            emb_dim: number of embedding dimensions of the time encoding.
        """

        super().__init__()
        
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_layer = ConvBlock(in_channels, in_channels * 2, num_groups, emb_dim)

    def forward(self, x, t):
        return self.conv_layer(self.down(x), t)


class Up(nn.Module):

    def __init__(self, in_channels, num_groups, emb_dim):
        """
        Implement an upsampling layer and a ConvBlock. The upsampling layer
        also contains a residual input from the corresponding downsampling
        layer.

        Args:
            in_channels: number of channels of the input feature map.
            num_groups: number of groups in the GroupNorm.
            emb_dim: number of embedding dimensions of the time encoding.
        """

        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False)
        self.conv_layer = ConvBlock(in_channels, in_channels // 2, num_groups, emb_dim)

    def forward(self, x, x_res, t):

        x = self.up(x)

        # If the shapes for the residual connection don't match, pad the smaller feature.
        if x.shape[2:] != x_res.shape[2:]:

            diffH = x_res.shape[2] - x.shape[2]
            diffW = x_res.shape[3] - x.shape[3]

            x = F.pad(x, (diffW//2, diffW - diffW//2, diffH//2, diffH - diffH//2))

        out = torch.cat([x, x_res], dim=1)
        out = self.conv_layer(out, t)

        return out


class TimeEncoding(nn.Module):

    def __init__(self, dim):
        """
        Args:
            dim: the number of dimensions which represent the encoding.
        """

        super().__init__()

        assert dim % 2 == 0 and dim != 2, f"The encoding dimension must be divisible by 2 and greater than 3, but dim={dim} was given."
        half_dim = dim // 2
        # In denom we specify the dimension indices to be 0 ,..., half_dim-1.
        # So, in const we set the denominator to be half_dim-1.
        const = math.log(1e4) / (half_dim - 1)
        self.register_buffer("denom", torch.exp(-torch.arange(half_dim) * const))

    def forward(self, t):
        angle = t[:, None].to(self.denom.dtype) @ self.denom[None, :]
        encodings = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1).view(t.shape[0], -1)
        return encodings


class UNet(nn.Module):

    def __init__(self, input_channels, hid_channels, num_blocks, emb_dim):
        """
        Args:
            in_channels: number of channels of the input image.
            hid_channels: number of channels of the feature map after the first conv layer.
            num_blocks: number of downsampling (or upsampling) layers.
            emb_dim: number of embedding dimensions of the time encoding.
        """

        super().__init__()

        # The number of groups to be used for the groupnorm is set to hid_channels.
        num_groups = hid_channels

        self.num_blocks = num_blocks
        self.time_enc = nn.Sequential(
            TimeEncoding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )
        self.input_layer = ConvBlock(input_channels, hid_channels, num_groups, emb_dim)

        dims = [(1 << i) for i in range(self.num_blocks)]

        self.down = nn.ModuleList([Down(hid_channels * dim, num_groups, emb_dim) for dim in dims])
        self.up = nn.ModuleList([Up(hid_channels * (1 << self.num_blocks) // dim, num_groups, emb_dim) for dim in dims])
        self.output_layer = nn.Conv2d(hid_channels, input_channels, kernel_size=1, bias=False)

    def forward(self, x, t):

        assert x.shape[-1] // (1 << self.num_blocks) > 1, \
        f"The depth of the UNet is large for the image size, try num_blocks < {int(math.log(x.shape[-1]) / math.log(2))}."

        assert x.shape[0] == t.shape[0], f"The input image and timesteps should have the same batch size, {x.shape[0]} != {t.shape[0]}."

        t = self.time_enc(t)
        x = self.input_layer(x, t)

        residuals = []

        for i in range(self.num_blocks):
            residuals.append(x)
            x = self.down[i](x, t)

        for i in range(self.num_blocks):
            x = self.up[i](x, residuals[-i-1], t)

        x = self.output_layer(x)

        return x


def main():

    time_enc = TimeEncoding(1024)
    t = torch.arange(1000)
    out = time_enc(t)
    plt.imshow(out)
    plt.show()

    return



if __name__ == "__main__":
    main()
