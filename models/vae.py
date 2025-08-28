"""
VAE (Variational Autoencoder) 编解码器
用于将图像编码到潜在空间和从潜在空间解码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """VAE 中的残差块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        
        return x + residual


class AttentionBlock(nn.Module):
    """自注意力块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        
        self.scale = channels ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        norm_x = self.norm(x)
        
        # 计算 QKV
        q = self.q(norm_x).view(B, C, H*W).transpose(1, 2)
        k = self.k(norm_x).view(B, C, H*W).transpose(1, 2)
        v = self.v(norm_x).view(B, C, H*W).transpose(1, 2)
        
        # 注意力计算
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.out(out)
        
        return x + out


class Encoder(nn.Module):
    """VAE 编码器"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        channels_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        use_attention: bool = True
    ):
        super().__init__()
        
        channels = 128
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        
        # 编码器块
        self.blocks = nn.ModuleList()
        ch_in = channels
        
        for i, mult in enumerate(channels_mult):
            ch_out = channels * mult
            
            for _ in range(num_res_blocks):
                self.blocks.append(ResidualBlock(ch_in, ch_out))
                ch_in = ch_out
                
            # 添加注意力层（在中间分辨率）
            if use_attention and i == len(channels_mult) // 2:
                self.blocks.append(AttentionBlock(ch_out))
            
            # 下采样（除了最后一层）
            if i < len(channels_mult) - 1:
                self.blocks.append(
                    nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1)
                )
        
        # 最终层
        self.norm_out = nn.GroupNorm(32, ch_out)
        self.conv_out = nn.Conv2d(ch_out, latent_channels * 2, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Returns:
            mean: 潜在空间均值
            log_var: 潜在空间对数方差
        """
        x = self.conv_in(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        # 分割均值和对数方差
        mean, log_var = torch.chunk(x, 2, dim=1)
        
        return mean, log_var


class Decoder(nn.Module):
    """VAE 解码器"""
    
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        channels_mult: Tuple[int, ...] = (4, 4, 2, 1),
        num_res_blocks: int = 2,
        use_attention: bool = True
    ):
        super().__init__()
        
        channels = 128
        ch_in = channels * channels_mult[0]
        
        # 输入层
        self.conv_in = nn.Conv2d(latent_channels, ch_in, 3, padding=1)
        
        # 解码器块
        self.blocks = nn.ModuleList()
        
        for i, mult in enumerate(channels_mult):
            ch_out = channels * mult
            
            for _ in range(num_res_blocks):
                self.blocks.append(ResidualBlock(ch_in, ch_out))
                ch_in = ch_out
            
            # 添加注意力层
            if use_attention and i == len(channels_mult) // 2:
                self.blocks.append(AttentionBlock(ch_out))
            
            # 上采样（除了最后一层）
            if i < len(channels_mult) - 1:
                self.blocks.append(
                    nn.ConvTranspose2d(ch_out, ch_out, 4, stride=2, padding=1)
                )
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, ch_out)
        self.conv_out = nn.Conv2d(ch_out, out_channels, 3, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码"""
        x = self.conv_in(z)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x


class VAE(nn.Module):
    """完整的 VAE 模型"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        channels_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.encoder = Encoder(
            in_channels, latent_channels, 
            channels_mult, num_res_blocks, use_attention
        )
        
        self.decoder = Decoder(
            in_channels, latent_channels,
            tuple(reversed(channels_mult)), 
            num_res_blocks, use_attention
        )
        
        # 用于缩放潜在空间
        self.scale_factor = 0.18215
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码图像到潜在空间"""
        mean, log_var = self.encoder(x)
        mean = mean * self.scale_factor
        log_var = log_var * self.scale_factor
        return mean, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码图像"""
        z = z / self.scale_factor
        return self.decoder(z)
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整的前向传播
        Returns:
            recon_x: 重建图像
            mean: 潜在均值
            log_var: 潜在对数方差
        """
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        
        return recon_x, mean, log_var
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """从先验分布采样生成图像"""
        z = torch.randn(num_samples, 4, 32, 32).to(device)
        samples = self.decode(z)
        return samples
    
    def loss_function(
        self, 
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mean: torch.Tensor, 
        log_var: torch.Tensor,
        kld_weight: float = 1e-3
    ) -> dict:
        """
        计算 VAE 损失
        """
        # 重建损失
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL 散度损失
        kld_loss = -0.5 * torch.mean(
            1 + log_var - mean.pow(2) - log_var.exp()
        )
        
        # 总损失
        total_loss = recon_loss + kld_weight * kld_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss
        }


if __name__ == "__main__":
    # 测试 VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vae = VAE(
        in_channels=3,
        latent_channels=4,
        channels_mult=(1, 2, 4, 4)
    ).to(device)
    
    # 测试编码和解码
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        # 编码
        mean, log_var = vae.encode(x)
        print(f"Latent shape: {mean.shape}")
        
        # 解码
        z = vae.reparameterize(mean, log_var)
        recon = vae.decode(z)
        print(f"Reconstructed shape: {recon.shape}")
        
        # 完整前向传播
        recon_x, mean, log_var = vae(x)
        
        # 计算损失
        losses = vae.loss_function(recon_x, x, mean, log_var)
        print(f"Losses: {losses}")