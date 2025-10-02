import torch
import torch.nn as nn
import torch.nn.functional as F

# Vanilla
class KnotsModelVanila(nn.Module):
    def __init__(self, input_shape: int, hidden_units_1: int, hidden_units_2: int, hidden_units_3:int, hidden_units_4: int, output_shape: int):
        super().__init__()
        self.dropout = nn.Dropout(p=0.9)

        self.layer_prepare = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(input_shape)
        )
        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units_1),
            nn.ELU(),
            nn.BatchNorm1d(hidden_units_1)
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=hidden_units_1, out_features=hidden_units_2),
            nn.ELU(),
            nn.BatchNorm1d(hidden_units_2)
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=hidden_units_2, out_features=hidden_units_3),
            nn.ELU(),
            nn.BatchNorm1d(hidden_units_3)
        )
        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=hidden_units_3, out_features=hidden_units_4),
            nn.ELU()
        )
        self.layer_5 = nn.Sequential(
            nn.Linear(in_features=hidden_units_4, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
      x = self.layer_prepare(x)
      x = self.layer_1(x)
      x = self.layer_2(x)
      x = self.layer_3(x)
      x = self.layer_4(x)
      x = self.dropout(x)
      x = self.layer_5(x)
      return x

# CNN
class KnotsModelCNN(nn.Module):
    def __init__(self, output_shape: int):
        super().__init__()

        self.conv_1 = nn.Sequential(
          nn.Conv2d(1, 4, kernel_size=11, stride=1, dilation=2, padding=0),
          nn.BatchNorm2d(4),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(4, 16, kernel_size=5, stride=1, dilation=2, padding=0),
          nn.BatchNorm2d(16),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(16, 64, kernel_size=3, stride=1, dilation=2, padding=0),
          nn.BatchNorm2d(64),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(64, 256, kernel_size=3, stride=1, dilation=2, padding=0),
          nn.BatchNorm2d(256),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(256, 361, kernel_size=3, stride=1, padding=0),
          nn.BatchNorm2d(361),
          nn.ReLU(),

          nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(361*(12)**2),

            nn.Dropout(p=0.7),

            nn.Linear(in_features=361*(12)**2,
                      out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Dropout(p=0.7),

            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.ReLU(),

            nn.Linear(4096, output_shape)
        )

    def forward(self, x: torch.Tensor):
      x = self.conv_1(x)
      x = self.classifier(x)
      return x

# CvT
class ConvEmbedding(nn.Module):
  def __init__(self, patch_size, stride, in_dim, embed_dim):
    super().__init__()

    self.embed_conv = nn.Sequential(
        nn.Conv2d(in_dim, embed_dim, kernel_size=patch_size, stride=stride),
        nn.GELU())
    self.norm = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(0.1)

    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Conv2d):
      init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      init.xavier_uniform_(m.weight)
      if m.bias is not None:
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      init.constant_(m.weight, 1.0)
      init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
      init.constant_(m.weight, 1.0)
      init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.embed_conv(x) # [B, in_dim, H, W] -> [B, embed_dim, H//stride, W//stride] = [B, C, H', W']
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2) # [B, C, H', W'] -> [B, C, H'*W'] -> [B, H'*W', C]
    x = self.norm(x)
    return x

class ConvProj(nn.Module):
  def __init__(self, in_dim, embed_dim, num_heads):
    super().__init__()
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_dim = embed_dim//num_heads
    self.embed_dim = embed_dim

    self.q_proj = nn.Sequential(
        nn.Conv2d(in_dim, embed_dim, kernel_size=3, padding=1, stride=1, groups=in_dim),
        nn.GELU(),
        nn.BatchNorm2d(embed_dim),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        nn.GELU()
    )

    self.k_proj = nn.Sequential(
        nn.Conv2d(in_dim, embed_dim, kernel_size=3, padding=1, stride=2, groups=in_dim),
        nn.GELU(),
        nn.BatchNorm2d(embed_dim),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        nn.GELU()
    )

    self.v_proj = nn.Sequential(
        nn.Conv2d(in_dim, embed_dim, kernel_size=3, padding=1, stride=2, groups=in_dim),
        nn.GELU(),
        nn.BatchNorm2d(embed_dim),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        nn.GELU()
    )

    self.dropout = nn.Dropout(0.1)

    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Conv2d):
      init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      init.xavier_uniform_(m.weight)
      if m.bias is not None:
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      init.constant_(m.weight, 1.0)
      init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
      init.constant_(m.weight, 1.0)
      init.constant_(m.bias, 0)

  def reshape_head(self, x):
    B, C, H, W = x.shape
    x = x.reshape(B, self.num_heads, self.head_dim, H*W) # [B, C, H, W] -> [B, num_heads, head_dim, H*W]
    return x.permute(0, 3, 1, 2) # [B, num_heads, head_dim, H*W] -> [B, H*W, num_heads, head_dim]

  def forward(self, x):
    B, HW, C = x.shape
    H = int(math.sqrt(HW))
    assert H**2 == HW, "HW should be a square number"

    x = x.permute(0, 2, 1) # [B, HW, C] -> [B, C, HW]
    x = x.reshape(B, C, H, H) # [B, C, HW] -> [B, C, H, W]

    q = self.q_proj(x) # [B, C, H, W] -> [B, embed_dim, H, W]
    k = self.k_proj(x) # [B, C, H, W] -> [B, embed_dim, H//2, W//2]
    v = self.v_proj(x) # [B, C, H, W] -> [B, embed_dim, H//2, W//2]

    q = self.reshape_head(q) # [B, embed_dim, H, W] = [B, num_heads*head_dim, H, W] -> [B, num_heads, head_dim, H*W] -> [B, H*W, num_heads, head_dim]
    k = self.reshape_head(k) # [B, embed_dim, H//2, W//2] -> [B, H//2*W//2, num_heads, head_dim]
    v = self.reshape_head(v) # [B, embed_dim, H//2, W//2] -> [B, H//2*W//2, num_heads, head_dim]

    return q, k, v

class ConvTransformerBlock(nn.Module):
  def __init__(self, in_dim, embed_dim, num_heads, mlp_dim): # in_dim = C
    super().__init__()

    self.proj = ConvProj(in_dim=in_dim, embed_dim=embed_dim, num_heads=num_heads)
    self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    self.mlp = nn.Sequential(
        nn.LayerNorm(in_dim+embed_dim),
        nn.Linear(in_dim+embed_dim, mlp_dim),
        nn.GELU(),
        nn.Linear(mlp_dim, in_dim+embed_dim),
        nn.GELU()
    )

    self.lin_reshape = nn.Sequential(
        nn.Linear(in_dim+embed_dim, embed_dim),
        nn.GELU()
    )

    self.attn_dropout = nn.Dropout(0.1)
    self.mlp_dropout = nn.Dropout(0.1)
    self.lin_dropout = nn.Dropout(0.1)

    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Conv2d):
      init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      init.xavier_uniform_(m.weight)
      if m.bias is not None:
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      init.constant_(m.weight, 1.0)
      init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
      init.constant_(m.weight, 1.0)
      init.constant_(m.bias, 0)
  def forward(self, x):
    # x.shape = [B, H*W, C]

    q, k, v = self.proj(x)

    B, seq_length_q, _, _ = q.shape # seq_length_q=H*W
    _, seq_length_kv, _, _ = k.shape # seq_length_kv=H//2*W//2

    q_flat = q.reshape(B, seq_length_q, -1) # [B, seq_length_q, num_heads, head_dim] -> [B, seq_length_q, embed_dim]
    k_flat = k.reshape(B, seq_length_kv, -1) # [B, seq_length_k, num_heads, head_dim] -> [B, seq_length_k, embed_dim]
    v_flat = v.reshape(B, seq_length_kv, -1) # [B, seq_length_v, num_heads, head_dim] -> [B, seq_length_v, embed_dim]

    attn, _ = self.mha(q_flat, k_flat, v_flat) # [B, seq_length_q, embed_dim] = [B, H*W, embed_dim]

    out_attn = torch.cat([x, attn], dim=-1) # [B, H*W, embed_dim+C]

    out_mlp = self.mlp(out_attn) # [B, H*W, embed_dim+C]

    out = self.lin_reshape(out_attn + out_mlp) # [B, H*W, embed_dim]

    return out

class CvTStage(nn.Module):
  def __init__(self, patch_size, stride, in_dim, embed_dim, num_heads, mlp_dim, out_dim, N):
    super().__init__()

    self.emb = ConvEmbedding(patch_size=patch_size, stride=stride, in_dim=in_dim, embed_dim=embed_dim)
    self.trblocks = nn.ModuleList()
    for i in range(N-1):
      self.trblocks.append(ConvTransformerBlock(in_dim=embed_dim, embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim))
    self.trblocks.append(ConvTransformerBlock(in_dim=embed_dim, embed_dim=out_dim, num_heads=num_heads, mlp_dim=mlp_dim))

  def forward(self, x):
    # x.shape = [B, C, H, W]
    x = self.emb(x)
    for trblock in self.trblocks:
      x = trblock(x)

    B, HW, out_dim = x.shape
    H = int(math.sqrt(HW))
    x = x.permute(0, 2, 1) # [B, out_dim, H*W]
    x = x.reshape(B, out_dim, H, H) # [B, out_dim, H, W]
    return x

class KnotsModelCvT(nn.Module):
  def __init__(self,
               Ns=(1, 1, 1),
               params=(
                   (8, 8, 1, 64, 2, 128, 64),
                   (4, 4, 64, 64, 2, 128, 64),
                   (2, 2, 64, 64, 2, 128, 64)
               )):
    super().__init__()

    self.stages = nn.ModuleList()
    for stage, (param, N) in enumerate(zip(params, Ns)):
      patch_size, stride, in_dim, embed_dim, num_heads, mlp_dim, out_dim = param
      self.stages.append(
          CvTStage(patch_size, stride, in_dim, embed_dim, num_heads, mlp_dim, out_dim, N)
      )

    self.weighted_sum = nn.Sequential(
        nn.Flatten(),

        nn.Linear(params[-1][-1]*(1)**2, 1),
        nn.ReLU()
    )

    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    for stage in self.stages:
      x = stage(x)

    x = self.weighted_sum(x)
    x = x.mean(dim=1).unsqueeze(dim=1)

    return x
