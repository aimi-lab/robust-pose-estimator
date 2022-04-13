from torch import nn
import torch
import math
from segmentation_models_pytorch import DeepLabV3Plus
from typing import Iterable


def att_weight_plot(att_weights, hw):
    # attention flow field
    att1 = att_weights[:, -att_weights.shape[0]:]
    att_max = torch.argmax(att1, dim=1)
    att_max = att_max.reshape(hw)
    att_max_w = att_max / hw[0]
    att_max_h = att_max % hw[0]
    import matplotlib.pyplot as plt
    #fig , ax = plt.subplots(1,2)
    plt.matshow(att_weights.numpy())
    #ax[1].imshow(att_max_h.numpy())
    plt.savefig('att_max.png')
    plt.close()
    return att_max_w, att_max_h


class PositionalEncoding(nn.Module):
    def __init__(self, size, max_sequence_len=8):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_sequence_len, 1,1, *size))

    def forward(self, x):
        s, b = x.shape[:2]
        x += self.pos_embedding[-s:]
        return x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ProjectionHead, self).__init__()
        self.conv = nn.Conv2d(input_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TMABlock(nn.Module):
    # non-local block as in Temporal Memory Attention for Video Semantic Segmentation
    def __init__(self, input_dim, input_size):
        super(TMABlock, self).__init__()
        latent_dim = int(input_dim/4)
        self.proj_q = ProjectionHead(input_dim, latent_dim)
        self.proj_k = ProjectionHead(input_dim, latent_dim)
        self.proj_v = ProjectionHead(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.latent_dim = latent_dim
        self.scale = 1 / math.sqrt(latent_dim)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, query, key, value=None, mask=None):
        '''
        Parameters:
       -----------
           query: torch.tensor, shape(batch, F)
           key: torch.tensor, shape(N, batch, F)
           value: torch.tensor, shape(N, batch, F)
               if no value is provided use key as value
           mask: torch.tensor, shape(Nt, N)
               no peak mask, 0.0 -> valid, -inf -> ignore weight
       '''
        s,b,c,w,h = key.shape
        if value is None:
            value = key

        query_r = self.norm(query)
        key = self.norm(key)
        value = self.norm(value)
        # input projections
        query_r = self.proj_q(query_r).view(b, self.latent_dim, w*h).permute(0, 2, 1)
        key = self.proj_k(key.view(s*b,c,w,h)).view(s,b,self.latent_dim,w*h).permute(1, 2, 0, 3).reshape(b, self.latent_dim, s*w*h)
        value = self.proj_v(value.view(s*b,c,w,h)).view(s,b,c,w*h).permute(1, 0, 3, 2).reshape(b, s*w*h, c)

        # attention weights
        att_weight = torch.bmm(query_r, key)
        # apply mask -> normalize + softmax
        if mask is not None:
            att_weight = att_weight + mask
        att_weight = self.softmax(att_weight * self.scale)
        # applying weights to value
        y = torch.bmm(att_weight, value)
        # residual connection
        y = y.view(b, c, w, h) + query
        return y, att_weight


class DeepLabTAM(nn.Module):
    """
    DeepLabV3+ with TAM in Bottleneck
    """
    def __init__(self, segmentation_model: DeepLabV3Plus, input_size: Iterable): #, latent_dim: int, dropout: float=0.2):
        super(DeepLabTAM, self).__init__()
        self.segmentation_model = segmentation_model
        assert len(input_size) == 2
        input_size = [int(d / 16) for d in input_size]
        self.tam = TMABlock(segmentation_model.encoder.out_channels[-1], input_size)

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x : input tensor shape(B,S,C,W,H)

        Returns
        -------

        """
        # Backbone Encoder
        x = x.permute(1,0,2,3,4)  # reshape to sequence first (S,B,C,W,H)
        s,b,c,w,h = x.shape
        features = self.segmentation_model.encoder(x.reshape(s*b, c, w, h))
        c, w, h = features[-1].shape[-3:]
        # add TAM in Bottleneck
        bottle_neck_features = features[-1].view(s,b,c,w,h)
        query = bottle_neck_features[-1]
        key_value = bottle_neck_features
        bottle_neck_features, _ = self.tam(query, key_value)
        features[-1] = bottle_neck_features
        if len(features) == 6:
            features[-4] = features[-4][-b:]  # EfficientNet
        else:
            features[-3] = features[-3][-b:]  # PVT
        # DeepLabV3+ Decoder
        decoder_output = self.segmentation_model.decoder(*features)

        masks = self.segmentation_model.segmentation_head(decoder_output)

        return masks

