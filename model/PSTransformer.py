import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.LN = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        hurst_list = []
        sigma_list = []
        tau_list = []

        for i, attn_layer in enumerate(self.attn_layers):
            x, series, prior, sigma, hurst, soomthness_loss, beta_prior_loss = attn_layer(x, x, x, attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            hurst_list.append(hurst)
            sigma_list.append(sigma)
            tau_list.append(tau)

