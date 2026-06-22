import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LeadAwareStem(nn.Module):
    """Decoupled Spatial-Temporal Stem for multi-lead ECG signals."""
    def __init__(self, in_ch=12, base_dim=64, kernel_size=15, stride=2):
        super().__init__()
        self.features_per_lead = 8
        hidden_dim = in_ch * self.features_per_lead

        # Depthwise convolution for lead-independent temporal feature extraction
        self.lead_conv = nn.Conv1d(in_ch, hidden_dim, kernel_size=kernel_size,
                                   stride=stride, padding=kernel_size // 2,
                                   groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()

        # Pointwise convolution for cross-lead feature fusion
        self.lead_mix = nn.Conv1d(hidden_dim, base_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(base_dim)

    def forward(self, x):
        x = self.act(self.bn1(self.lead_conv(x)))
        x = self.bn2(self.lead_mix(x))
        return x

class AdaptiveSpectrumFilter(nn.Module):
    """FFT-based filter to adaptively recalibrate frequency components."""
    def __init__(self, dim, num_bands=32):
        super().__init__()
        self.num_bands = num_bands
        self.register_buffer('window', None, persistent=False)

        self.freq_gate = nn.Sequential(
            nn.Conv1d(dim, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim // 4),
            nn.ReLU(),
            nn.Conv1d(dim // 4, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, L = x.shape
        orig_dtype = x.dtype

        if self.window is None or self.window.shape[-1] != L:
            win = torch.hann_window(L, periodic=True, device=x.device)
            self.window = win.view(1, 1, L)

        # STFT-like processing using Hann window and RFFT
        x_f32 = x.float() * self.window
        x_fft = torch.fft.rfft(x_f32, dim=-1, norm='ortho')
        x_mag = torch.abs(x_fft)
        x_log = torch.log(x_mag + 1e-6)

        # Generate frequency-domain gates via adaptive pooling
        x_pooled = F.adaptive_avg_pool1d(x_log, self.num_bands)
        gate = self.freq_gate(x_pooled.to(orig_dtype))
        gate_up = F.interpolate(gate, size=x_fft.shape[-1], mode='linear', align_corners=False)

        x_fft_filtered = x_fft * gate_up.float()
        x_out = torch.fft.irfft(x_fft_filtered, n=L, dim=-1, norm='ortho')

        return x + x_out.to(orig_dtype)

class GRN(nn.Module):
    """Global Response Normalization (from ConvNeXt-V2)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x

class ModernECGBlock(nn.Module):
    """ConvNeXt-style block adapted for 1D ECG signals."""
    def __init__(self, dim, kernel_size=31, drop_path=0.0, use_fft=False):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(0.2)

        self.use_fft = use_fft
        self.asf = AdaptiveSpectrumFilter(dim, num_bands=16) if use_fft else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = x.permute(0, 2, 1) # Channels-last for LayerNorm and Linear layers
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)

        if self.asf is not None:
            x = self.asf(x)

        return shortcut + self.drop_path(x)

class ConditionalPositionalEncoding(nn.Module):
    """CPE to handle variable length or provide local context."""
    def __init__(self, dim, kernel_size=15):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + self.proj(x)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mask = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * mask

class RefinedCrossScaleViT(nn.Module):
    """Cross-scale transformer to bridge shallow (high-res) and deep (low-res) features."""
    def __init__(self, dim, shallow_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.saliency_gate = SpatialAttention1D(kernel_size=7)
        self.kv_proj_conv = nn.Conv1d(shallow_dim, dim, kernel_size=8, stride=8)
        self.pos_cpe_q = ConditionalPositionalEncoding(dim, kernel_size=15)
        self.pos_cpe_k = ConditionalPositionalEncoding(dim, kernel_size=15)

        self.norm_q, self.norm_k, self.norm_v = [nn.LayerNorm(dim) for _ in range(3)]
        self.q_proj, self.k_proj, self.v_proj = [nn.Linear(dim, dim) for _ in range(3)]
        self.out_proj = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 4, dim)
        )

    def forward(self, x_deep, x_shallow):
        B, C, L_d = x_deep.shape
        # Process shallow features as KV
        x_shallow = self.saliency_gate(x_shallow)
        x_shallow_tok = self.pos_cpe_k(self.kv_proj_conv(x_shallow))
        x_deep = self.pos_cpe_q(x_deep)

        q = self.norm_q(x_deep.permute(0, 2, 1))
        k = self.norm_k(x_shallow_tok.permute(0, 2, 1))
        v = self.norm_v(x_shallow_tok.permute(0, 2, 1))

        # Multi-head attention implementation
        q = self.q_proj(q).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).reshape(B, L_d, C)
        x = q.transpose(1, 2).reshape(B, L_d, C) + self.out_proj(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x.permute(0, 2, 1)

class ContextGatedFusion(nn.Module):
    """Dynamic multi-scale feature integration using channel gates."""
    def __init__(self, dims, out_dim=128):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(d, out_dim, 1, bias=False), nn.BatchNorm1d(out_dim), nn.ReLU())
            for d in dims
        ])
        total_dim = out_dim * len(dims)
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(total_dim, total_dim // 4, 1), nn.ReLU(),
            nn.Conv1d(total_dim // 4, total_dim, 1), nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(total_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim), nn.SiLU(inplace=True)
        )

    def forward(self, features):
        target_size = features[0].shape[-1]
        processed = []
        for i, feat in enumerate(features):
            x = self.projs[i](feat)
            if x.shape[-1] != target_size:
                x = F.interpolate(x, size=target_size, mode='linear', align_corners=False)
            processed.append(x)
        x_cat = torch.cat(processed, dim=1)
        return self.fusion_conv(x_cat * self.gate_generator(x_cat))

class Query2Label_Head(nn.Module):
    """Semantic Gated Head using label queries and Transformer Decoders."""
    def __init__(self, num_classes, dim, nhead=8, num_layers=2, text_embed_path=None):
        super().__init__()
        self.use_text_prior = False
        self.query_bias = nn.Parameter(torch.zeros(num_classes, dim))

        if text_embed_path is not None:
            try:
                loaded_embeds = torch.load(text_embed_path, map_location='cpu')
                if loaded_embeds.shape[0] == num_classes:
                    self.use_text_prior = True
                    self.register_buffer('text_anchors', loaded_embeds)
                    self.text_proj = nn.Linear(loaded_embeds.shape[1], dim, bias=False)
                    self.text_norm = nn.LayerNorm(dim)
                    print(f"[Query2Label] Text Priors Loaded.")
            except Exception as e:
                print(f"[Error] Text Embedding Load Failed: {e}")

        if not self.use_text_prior:
            self.label_embed = nn.Embedding(num_classes, dim)
            nn.init.orthogonal_(self.label_embed.weight)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=dim, nhead=nhead, batch_first=True,
                dim_feedforward=dim * 4, dropout=0.1, norm_first=True
            ), num_layers=num_layers
        )
        self.memory_norm = nn.LayerNorm(dim)
        self.norm_final = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(dim, 1)
        self.pos_cpe = ConditionalPositionalEncoding(dim, kernel_size=15)

    def forward(self, x):
        x = self.pos_cpe(x)
        memory = self.memory_norm(x.permute(0, 2, 1))

        if self.use_text_prior:
            tgt = (self.text_norm(self.text_proj(self.text_anchors)) + self.query_bias).unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            tgt = self.label_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)

        out = self.decoder(tgt=tgt, memory=memory)
        return self.fc(self.dropout(self.norm_final(out))).squeeze(-1)

class LayerNormChannelsFirst(nn.Module):
    """LayerNorm for (Batch, Channel, Length) format."""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None] * x + self.bias[:, None]

class MSS_Net(nn.Module):
    """Multi-scale Residual Gated Network (Main Model)."""
    def __init__(self, nOUT, in_ch=12, base_dim=64, drop_path_rate=0.2,
                 text_embed_path='./cpsc2018_9class_embeddings.pt',
                 ablation_mode='full'):
        super(MSS_Net, self).__init__()
        self.ablation_mode = ablation_mode
        self.inconv = LeadAwareStem(in_ch=in_ch, base_dim=base_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6)]

        # Encoder stages
        self.stage1 = nn.Sequential(ModernECGBlock(base_dim, dpr[0]), ModernECGBlock(base_dim, dpr[1]))
        self.down1 = nn.Sequential(LayerNormChannelsFirst(base_dim), nn.Conv1d(base_dim, base_dim * 2, 2, 2))

        self.stage2 = nn.Sequential(ModernECGBlock(base_dim * 2, dpr[2]), ModernECGBlock(base_dim * 2, dpr[3]))
        self.down2 = nn.Sequential(LayerNormChannelsFirst(base_dim * 2), nn.Conv1d(base_dim * 2, base_dim * 4, 2, 2))

        # Adaptive Spectrum Filter (ASF) enabled in Stage 3 for non-baseline modes
        use_fft_s3 = ablation_mode != 'baseline'
        self.stage3 = nn.Sequential(
            ModernECGBlock(base_dim * 4, dpr[4], use_fft=use_fft_s3),
            ModernECGBlock(base_dim * 4, dpr[5], use_fft=use_fft_s3)
        )

        # BGF Module (Bridge & Fusion)
        if ablation_mode in ['bgf', 'full']:
            self.bridge = RefinedCrossScaleViT(dim=base_dim * 4, shallow_dim=base_dim * 2)
            self.fusion = ContextGatedFusion(dims=[base_dim, base_dim * 2, base_dim * 4], out_dim=128)
        else:
            self.simple_adapter = nn.Sequential(
                nn.Conv1d(base_dim * 4, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU()
            )

        # SGH Module (Query2Label head)
        self.q2l_head = Query2Label_Head(nOUT, 128, text_embed_path=(text_embed_path if ablation_mode == 'full' else None))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] != 12: x = x.permute(0, 2, 1)

        x0 = self.inconv(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(self.down1(x1))
        x3 = self.stage3(self.down2(x2))

        # Ablation-specific fusion logic
        if self.ablation_mode in ['bgf', 'full']:
            x3_enhanced = self.bridge(x_deep=x3, x_shallow=x2)
            out_feat = self.fusion([x1, x2, x3_enhanced])
        else:
            out_feat = self.simple_adapter(x3)

        return self.q2l_head(out_feat)