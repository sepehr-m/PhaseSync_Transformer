import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMask:
    def __init__(self, B, L, device):
        mask_shape=[B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                    torch.ones(mask_shape, dtype=torch.bool), diagonal=1
                    ).to(device)
    @property
    def mask(self):
        return self._mask

class PhaseSyncAttention(nn.Module):
    def __init__(self,
                 win_size,
                 mask_flag=True,
                 scale=None,
                 attention_dropout=0.0,
                 output_attention=False,
                 gamma=2.0,
                 sigma=1.0,
                 lambda_smooth=0.01,
                 device='mps',
                 ):
        super(PhaseSyncAttention, self).__init__()
        self.device = torch.device(deivce)
        self.win_size = win_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.gamma = gamma
        self.sigma = sigma
        self.lambda_smooth = lambda_smooth
        
        indices = torch.arange(win_size, device=self.device)
        self.distances = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))

    def compute_fractal_abscissa(self, hurst):
        hurst = torch.clamp(hurst, min=0.01, max=1.0)
        exp_hurst = torch.exp(self.gamma * hurst)
        psi = torch.cumsum(exp_hurst, dim=1)
        return psi

    def compute_gaussian_prior(self, psi):
        B, L, H = psi.shape
        
        psi_i = psi.unsqueeze(2) # [B, L, 1, H]
        psi_j = psi.unsqueeze(1) # [B, 1, L, H]
        diff = psi_i - psi_j # [B, L, L, H]

        prior = torch.exp(-diff ** 2 / (2 * self.sigma ** 2))
        prior = prior.permute(0, 3, 1, 2) # [B, H, L, L]
        prior_sum = prior.sum(dim=-1, keepdim=True)
        prior = prior /  (prior_sum + 1e-6)

        return prior



