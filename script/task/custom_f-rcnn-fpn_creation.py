from torch import nn
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops

from torchdistill.models.custom.bottleneck.processor import get_bottleneck_processor

class BottleneckBaseEncoder(nn.Module):
    def __init__(self, encoder, compressor=None):
        super().__init__()
        self.encoder = encoder
        self.compressor = compressor

    def forward(self, x):
        z = self.encoder(x)
        if self.compressor is not None:
            z = self.compressor(z)


class BottleneckBaseDecoder(nn.Module):
    def __init__(self, decoder, decompressor=None):
        super().__init__()
        self.decoder = decoder
        self.decompressor = decompressor

    def forward(self, x):
        if self.decompressor is not None:
            z = self.decompressor(z)
        return self.decoder(z)


class Bottleneck4LargeResNetEncoder(BottleneckBaseEncoder):
    """
    Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks
    """
    def __init__(self, bottleneck_channel, compressor=None):
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Relu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 256, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        super().__init__(encoder=encoder compressor=compressor)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()

class Bottleneck4LargeResNetDecoder(BottleneckBaseDecoder):
    """
    Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks
    """
    def __init__(self, bottleneck_channel, decompressor=None):
        decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        super().__init__(decoder=decoder, decompressor=decompressor)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()

def custom_resnet_fpn_backbone(backbone_name, norm_layer=misc_nn_ops.FrozenBatchNorm2d, useTorchBackbone=False):
    layer1 = None
    compressor = None
    decompressor = None

    compressor = get_bottleneck_processor('SimpleQuantizer', 8)
    decompressor = get_bottleneck_processor('SimpleDequantizer', 8)
    
    if not useTorchBackbone:
        layer1 = Bottleneck4LargeResNetEncoder(1, compressor)
    else:
        layer1 = Bottleneck4LargeResNetDecoder(1, decompressor)


    prefix = 'custom_'
    start_idx = backbone_name.find(prefix) + len(prefix)
    org_backbone_name = backbone_name[start_idx:] if backbone_name.startswith(prefix) else backbone_name
    backbone = resnet.__dict__[org_backbone_name](
        pretrained= False #backbone_params_config.get('pretrained', False),
        norm_layer=norm_layer
    )
    if layer1 is not None:
        backbone.layer1 = layer1

    #trainable_layers = backbone_params_config.get('trainable_backbone_layers', 4)
    # select layers that wont be frozen
    #assert 0 <= trainable_layers <= 6
    #layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'bn1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        #if all([not name.startswith(layer) for layer in layers_to_train]):
        parameter.requires_grad_(False)

    returned_layers = backbone_params_config.get('returned_layers', [1, 2, 3, 4])
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)