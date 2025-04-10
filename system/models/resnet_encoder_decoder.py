from torch import nn
from system.modules import DecoderBN,ResNetEncoder


class ResNetEncoderDecoder(nn.Module):
    """ResNet作为编码器+解码器
    input shape is [B,3,H,W]
    first: extracts images features
    second: decodes and up sampling
    output: high resolution immediate features S and shape is [B,C,h,w]
    set h=H/2 and w=W/2
    """
    def __init__(self, num_layers=50,num_features=256,model_dim=32):
        super(ResNetEncoderDecoder, self).__init__()
        self.encoder = ResNetEncoder(num_layers=num_layers,
                                     pretrained=True,
                                     num_input_images=1)

        self.decoder = DecoderBN(num_features=num_features,
                                 num_classes= model_dim,
                                 bottleneck_features=2048)
    def forward(self, x, **kwargs):
        x = self.encoder(x)
        return self.decoder(x, **kwargs)




