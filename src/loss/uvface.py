
from torch.nn import Module, Parameter
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.adaface import AdaFace
from loss.cosface import CosFace

class UVFace(Module):

    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 t_alpha=0.01,
                 wrapped_head='adaface',
                 ):
        super().__init__()

        self.t_alpha = t_alpha
        self.eps = 1e-6

        self.register_buffer('batch_norm_mean', torch.ones(1)*20)
        self.register_buffer('batch_norm_std', torch.ones(1)*10)

        self.register_buffer('batch_fiqa_mean', torch.ones(1))
        self.register_buffer('batch_fiqa_std', torch.ones(1)*10)
        
        if wrapped_head == 'adaface':
            self.wrapped_cls_head = AdaFace(embedding_size, classnum)
        elif wrapped_head == 'cosface':
            self.wrapped_cls_head = CosFace(embedding_size, classnum)
        else:
            raise ValueError(f"Unsupported UVFace wrapped_head: {wrapped_head}")

    def forward(self, embbedings, labels, fiqa_scores=None):

        norms = embbedings.norm(p=2, dim=1)                 
        norms_clamped = norms.clamp(1e-3, 100.0)            
        norms_stats = norms_clamped.detach()               

        with torch.no_grad():
            mean_norm = norms_stats.mean()
            std_norm  = norms_stats.std(unbiased=False).clamp_min(self.eps)

            self.batch_norm_mean = mean_norm * self.t_alpha + (1 - self.t_alpha) * self.batch_norm_mean
            self.batch_norm_std  = std_norm  * self.t_alpha + (1 - self.t_alpha) * self.batch_norm_std


        if fiqa_scores is None:
            fiqa = norms_clamped.detach()
        else:
            fiqa = fiqa_scores.detach().view(-1).clamp(-1.0, 100.0)

        with torch.no_grad():
            mean_fiqa = fiqa.mean()
            std_fiqa  = fiqa.std(unbiased=False).clamp_min(self.eps)

            self.batch_fiqa_mean = mean_fiqa * self.t_alpha + (1 - self.t_alpha) * self.batch_fiqa_mean
            self.batch_fiqa_std  = std_fiqa  * self.t_alpha + (1 - self.t_alpha) * self.batch_fiqa_std

        z = (fiqa - self.batch_fiqa_mean) / self.batch_fiqa_std
        z = z.clamp(-5.0, 5.0) 

        target_norm = (z * self.batch_norm_std) + self.batch_norm_mean
        target_norm = target_norm.clamp(1e-3, 100.0).detach()    

        norm_loss = F.mse_loss(norms_clamped, target_norm)

        # classification head (unchanged)
        cls_out = self.wrapped_cls_head(embbedings, labels)
        logits = cls_out[0] if isinstance(cls_out, (tuple, list)) else cls_out

        return logits, norm_loss

        
