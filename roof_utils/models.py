"""U-Net decoder with a torchvision ResNet18 encoder and one roof logit per pixel."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True))

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class RoofUNet(nn.Module):
    """ResNet18 encoder; compact U-Net decoder using GroupNorm for small batches.

    Returns logits (B,1,H,W). Encoder BatchNorm statistics stay fixed by default,
    even when encoder gradients are enabled. Use ImageNet-normalized RGB inputs.
    """
    def __init__(self, pretrained=True, weights_dir=None, freeze_encoder_bn=True):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.pretrained = bool(pretrained)
        self.weights_name = 'ResNet18_Weights.IMAGENET1K_V1' if pretrained else 'random initialization'
        if pretrained:
            if weights_dir is None:
                raise ValueError('Set weights_dir explicitly, e.g. paths.models_dir / "pretrained".')
            Path(weights_dir).mkdir(parents=True, exist_ok=True)
            weights = ResNet18_Weights.IMAGENET1K_V1
            state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(weights_dir),
                                                      check_hash=True, progress=True, map_location='cpu')
            self.encoder.load_state_dict(state)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.ModuleDict({
            'up3': UpBlock(512, 256, 128),
            'up2': UpBlock(128, 128, 64),
            'up1': UpBlock(64, 64, 32),
            'up0': UpBlock(32, 64, 16),
            'full': UpBlock(16, 3, 16),
            'head': nn.Conv2d(16, 1, 1)})
        self.freeze_encoder_bn = freeze_encoder_bn
        self.train(self.training)

    def set_encoder_trainable(self, trainable):
        for p in self.encoder.parameters():
            p.requires_grad_(trainable)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        e = self.encoder
        stem = e.relu(e.bn1(e.conv1(x)))
        s1 = e.layer1(e.maxpool(stem))
        s2 = e.layer2(s1)
        s3 = e.layer3(s2)
        s4 = e.layer4(s3)
        d = self.decoder['up3'](s4, s3)
        d = self.decoder['up2'](d, s2)
        d = self.decoder['up1'](d, s1)
        d = self.decoder['up0'](d, stem)
        d = self.decoder['full'](d, x)
        return self.decoder['head'](d)


def build_model(paths, pretrained=True):
    return RoofUNet(pretrained=pretrained, weights_dir=paths.models_dir/'pretrained')
