import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math

class PositionalEncoding(nn.Module):
  def __init__(self, in_channels: int, max_length: int):
    super().__init__()
    position = torch.arange(max_length).unsqueeze(-1)
    div_term = torch.exp(torch.arange(0, in_channels, 2) * (-math.log(10000.0) / in_channels)).unsqueeze(0)
    pe = torch.zeros(1, max_length, in_channels)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    return x + self.pe

class InputEmbeddings(nn.Module):

  def __init__(self, patches_size:int,latent_size:int, batch_size:int):
    super().__init__()
    # self.input_data    = input_data
    self.patche_size         = patches_size
    self.latent_size         = latent_size
    self.batch_size          = batch_size
    self.input_size          = self.patche_size * self.patche_size*3

    self.class_token         = nn.Parameter(torch.randn(1, 1, self.latent_size))
    #self.positional_encoding = nn.Parameter(torch.rand(1, 1, self.latent_size))
    self.positional_encoding = PositionalEncoding(1, 31).to('cuda')
    self.layer_norm          = nn.LayerNorm(latent_size).to(torch.float32).to('cuda')
    self.liner_projetction = nn.Linear(150, self.latent_size).to(torch.float32).to('cuda')
    self.liner_projetction_1 = nn.Linear(42, self.latent_size).to(torch.float32).to('cuda')
    #self.liner_projetction_2 = nn.Linear(150, self.latent_size).to(torch.float32).to('cuda')
    self.liner_projetction_3 = nn.Linear(66, self.latent_size).to(torch.float32).to('cuda')

  def forward(self, x1,y1,z1):
    x1 = einops.rearrange(x1, 'b c (h h1) (w w1) -> b c (h h1 w w1)', h1 = self.patche_size, w1 =self.patche_size)
    y1 = einops.rearrange(y1, 'b c (h h1) (w w1) -> b c (h h1 w w1)', h1=self.patche_size, w1=self.patche_size)
    z1 = einops.rearrange(z1, 'b c (h h1) (w w1) -> b c (h h1 w w1)', h1=self.patche_size, w1=self.patche_size)
    p, n, m = x1.shape
    x1 = self.liner_projetction_1(x1)
    y1 = self.liner_projetction_1(y1)
    z1 = self.liner_projetction_3(z1)
    #x1 = self.liner_projetction_1(x1)
    #y1 = self.liner_projetction_1(y1)
    #z1 = self.liner_projetction_3(z1)
    class_token   = nn.Parameter(torch.randn(p, 1, self.latent_size)).to('cuda')
    x1  = torch.cat((x1, class_token), dim=1)
    x1  = self.positional_encoding.forward(x1)
    x1  = self.layer_norm(x1)
    y1  = torch.cat((y1, class_token), dim=1)
    y1  = self.positional_encoding.forward(y1)
    y1  = self.layer_norm(y1)
    z1  = torch.cat((z1, class_token), dim=1)
    z1  = self.positional_encoding.forward(z1)
    z1  = self.layer_norm(z1)
    return x1, y1, z1

class EncoderBlock(nn.Module):

  def __init__(self, laten_size:int, num_head:int, embdin_dim:int, dropout:int=0.1):
    super().__init__()
    self.laten_size = laten_size
    self.num_head   = num_head
    self.embdin_dim = embdin_dim
    self.droupout   = dropout
    self.norm       = nn.LayerNorm(self.embdin_dim)
    self.attn_blck  = nn.MultiheadAttention(self.embdin_dim, self.num_head, self.droupout)
    self.mlp        = nn.Sequential(
                            nn.GELU(),
                            nn.Dropout(self.droupout),
                            nn.Linear(self.laten_size, self.laten_size),
                            nn.Dropout(self.droupout)
                      )
    self.layer_norm = nn.LayerNorm(self.laten_size)

  def forward(self,x1,y1,z1):
    x1 = x1.to(torch.float32)
    x1 = self.norm(x1)
    x1 = torch.permute(x1, (1, 0, 2))
    y1 = y1.to(torch.float32)
    y1 = self.norm(y1)
    y1 = torch.permute(y1, (1, 0, 2))
    z1 = z1.to(torch.float32)
    z1 = self.norm(z1)
    z1 = torch.permute(z1, (1, 0, 2))
    x = x1 + y1 + z1
    #x = torch.cat((x1,y1,z1), dim=-1)
    attn = self.attn_blck(x,x,x)
    attn = attn[0]
    attn = x + attn
    attn_2 = self.layer_norm(attn)
    x = self.mlp(attn_2)
    x = x + attn
    x = torch.permute(x, (1, 0, 2))
    return x.to('cuda')


class ViTModel(nn.Module):

  def __init__(self, patch_size:int, number_block:int, batch_size:int, embeddin_dim:int, num_head:int, latent_space:int, num_class:int, dropout:int):

    super().__init__()
    self.number_block = number_block
    self.latent_space = latent_space
    self.num_class    = num_class
    self.dropout      = dropout
    self.dim_emb      = embeddin_dim
    self.num_head     = num_head
    self.batch_size   = batch_size
    self.patch_size   = patch_size
    self.encoder      = EncoderBlock(self.latent_space, self.num_head, self.dim_emb, self.dropout)
    self.input_embg   = InputEmbeddings(self.patch_size, self.latent_space, self.batch_size)
    self.mlp = nn.Sequential(
      nn.LayerNorm(self.latent_space),
      nn.Linear(self.latent_space, self.latent_space),
      nn.Linear(self.latent_space, self.num_class)
    )

  def forward(self, x1, y1, z1):
   # x =x.to(torch.float32)
    x = 0
    x1,y1,z1 = self.input_embg(x1,y1,z1)
    for _ in range(1, self.number_block):
      x =  self.encoder(x1,y1,z1)
    x = x[:,0]
    #x =  self.mlp(x)
    return x
