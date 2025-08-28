"""
轻量级 U-Net 架构实现
采用深度可分离卷积和高效注意力机制减少参数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SeparableConv2d(nn.Module):
    """深度可分离卷积，减少参数量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientAttention(nn.Module):
    """高效注意力机制，降低计算复杂度"""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # 使用线性注意力近似
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 线性注意力计算，O(n) 复杂度
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # 计算全局上下文
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = 1 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)
        attn = torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z)
        
        attn = attn.reshape(B, H, W, C)
        x = self.proj(attn)
        return x


class ResBlock(nn.Module):
    """残差块with时间嵌入"""
    def __init__(self, in_channels, out_channels, time_emb_dim=256, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        # 使用深度可分离卷积
        self.conv1 = SeparableConv2d(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.conv2 = SeparableConv2d(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # 时间嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 跳跃连接
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
            
        # 可选的注意力层
        if use_attention:
            self.attention = EfficientAttention(out_channels)
            
    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        h = h + self.time_mlp(t)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # 注意力机制
        if self.use_attention:
            B, C, H, W = h.shape
            h_att = h.permute(0, 2, 3, 1)  # B, H, W, C
            h_att = self.attention(h_att)
            h = h_att.permute(0, 3, 1, 2)  # B, C, H, W
        
        return h + self.skip(x)


class LightweightUNet(nn.Module):
    """轻量级 U-Net 主体架构"""
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=[64, 128, 256, 512],
        time_emb_dim=256,
        num_res_blocks=2
    ):
        super().__init__()
        
        # 时间位置编码
        self.time_embed = nn.Sequential(
            nn.Linear(channels[0], time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 输入卷积
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # 编码器
        self.encoder = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            layers = []
            for j in range(num_res_blocks):
                layers.append(ResBlock(
                    in_ch if j == 0 else out_ch,
                    out_ch,
                    time_emb_dim,
                    use_attention=(i >= len(channels) // 2)  # 深层使用注意力
                ))
            if i < len(channels) - 1:
                layers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            self.encoder.append(nn.ModuleList(layers))
            in_ch = out_ch
        
        # 中间块
        self.middle = ResBlock(channels[-1], channels[-1], time_emb_dim, use_attention=True)
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(channels))):
            out_ch = channels[i]
            layers = []
            
            # 上采样
            if i < len(channels) - 1:
                layers.append(nn.ConvTranspose2d(
                    channels[i+1] if i < len(channels)-1 else channels[-1],
                    out_ch, 4, stride=2, padding=1
                ))
            
            # 残差块
            for j in range(num_res_blocks):
                skip_ch = out_ch * 2 if j == 0 and i < len(channels)-1 else out_ch
                layers.append(ResBlock(
                    skip_ch,
                    out_ch,
                    time_emb_dim,
                    use_attention=(i >= len(channels) // 2)
                ))
            self.decoder.append(nn.ModuleList(layers))
        
        # 输出卷积
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1)
        )
        
    def get_time_embedding(self, t, channels):
        """生成时间步的正弦位置编码"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq.to(t.device))
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq.to(t.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t, condition=None):
        """
        前向传播
        Args:
            x: 输入图像 [B, C, H, W]
            t: 时间步 [B]
            condition: 可选的条件信息
        """
        # 时间嵌入
        t = self.get_time_embedding(t[:, None], self.time_embed[0].in_features)
        t = self.time_embed(t)
        
        # 输入卷积
        x = self.conv_in(x)
        
        # 编码器 + 存储跳跃连接
        skips = []
        h = x
        for i, layers in enumerate(self.encoder):
            for layer in layers[:-1]:  # 残差块
                h = layer(h, t)
            skips.append(h)
            if i < len(self.encoder) - 1:
                h = layers[-1](h)  # 下采样
        
        # 中间块
        h = self.middle(h, t)
        
        # 解码器
        for i, layers in enumerate(self.decoder):
            # 上采样
            if len(layers) > num_res_blocks:
                h = layers[0](h)
            
            # 跳跃连接
            if i < len(skips):
                h = torch.cat([h, skips[-(i+1)]], dim=1)
            
            # 残差块
            start_idx = 1 if len(layers) > num_res_blocks else 0
            for layer in layers[start_idx:]:
                h = layer(h, t)
        
        # 输出
        h = self.conv_out(h)
        
        return h
    
    def count_parameters(self):
        """统计模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'total_mb': total * 4 / 1024 / 1024,  # 假设 float32
        }


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = LightweightUNet(
        in_channels=3,
        out_channels=3,
        channels=[64, 128, 256, 512],
        time_emb_dim=256,
        num_res_blocks=2
    )
    
    # 打印参数量
    params = model.count_parameters()
    print(f"Model Parameters: {params['total']:,}")
    print(f"Model Size: {params['total_mb']:.2f} MB")
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    t = torch.tensor([100, 200]).float()
    
    with torch.no_grad():
        out = model(x, t)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
    
    # 测试推理速度
    import time
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = model(x, t)
        end = time.time()
        print(f"Average inference time: {(end-start)/10*1000:.2f} ms")