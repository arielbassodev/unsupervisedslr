import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys

from lsfb_transfo.transforms import augmentation

class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.backbone = base_encoder
        self.projection = nn.Sequential(
           # nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    def permute_first_last_frames(self, x,y, z):
        batch_size, seq_len, num_points, dim = x.shape
        num_frames = seq_len // 3
        # Vérifier que num_frames est valide
        # Générer des indices de permutation pour les premières et dernières frames
        permuted_indices_first = torch.stack([torch.randperm(num_frames) for _ in range(batch_size)])
        permuted_indices_last = torch.stack([torch.randperm(num_frames) for _ in range(batch_size)])

        # Appliquer la permutation aux premières et dernières frames
        x_permuted = x.clone()  # Travailler sur une copie de x
        y_permuted = y.clone()
        z_permuted = z.clone()
        for i in range(batch_size):
            # Permuter les premières frames
           x_permuted[i, :num_frames] = x[i, permuted_indices_first[i]]
           y_permuted[i, :num_frames] = y[i, permuted_indices_first[i]]
           z_permuted[i, :num_frames] = z[i, permuted_indices_first[i]]
            # Permuter les dernières frames
           #x_permuted[i, -num_frames:] = x[i, seq_len - num_frames + permuted_indices_last[i]]
           #y_permuted[i, -num_frames:] = y[i, seq_len - num_frames + permuted_indices_last[i]]
           #z_permuted[i, -num_frames:] = z[i, seq_len - num_frames + permuted_indices_last[i]]

        return x_permuted.to(torch.float32), y_permuted.to(torch.float32), z_permuted.to(torch.float32)

    

    def generate_pairs(self, x,y,z):
        x = x.to('cuda')
        y = y.to('cuda')
        z = z.to('cuda')
        x1,y1,z1 = self.permute_first_last_frames(x,y,z)
        x2,y2,z2 = self.permute_first_last_frames(x,y,z)
        #sign_1 = torch.cat((x1,y,z1), dim=2)
        #sign_2 = torch.cat((x2,y,z2), dim=2)
        return x1,y1,z1,x2,y2,z2
        
    def forward(self, left, right, pose):
        x1,y1,z1,x2,y2,z2 = self.generate_pairs(left, right, pose)
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        y1 = y1.to(torch.float32)
        y2 = y2.to(torch.float32)
        z1 = z1.to(torch.float32)
        z2 = z2.to(torch.float32)
        x1 = x1.to('cuda')
        x2 = x2.to('cuda')
        y1 = y1.to('cuda')
        y2 = y2.to('cuda')
        z1 = z1.to('cuda')
        z2 = z2.to('cuda')
        h1 = self.backbone(x1,y1,z1)
        h2 = self.backbone(x2,y2,z2)
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        return z1, z2

