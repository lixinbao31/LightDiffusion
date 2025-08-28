"""
扩散模型核心实现
包含前向扩散过程、反向去噪过程和 DDIM 采样器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List


class GaussianDiffusion(nn.Module):
    """高斯扩散模型"""
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        loss_type: str = "l2",
        device: str = "cuda"
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.device = device
        
        # 初始化噪声调度
        self.betas = self._get_beta_schedule(beta_schedule, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # 预计算用于采样的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 后验分布参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def _get_beta_schedule(self, schedule: str, timesteps: int) -> torch.Tensor:
        """获取 beta 调度"""
        if schedule == "linear":
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """前向扩散过程：向干净图像添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """计算训练损失"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, condition)
        
        if self.loss_type == "l1":
            loss = F.l1_loss(predicted_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(predicted_noise, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """从噪声预测干净图像"""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """计算后验分布 q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(
        self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None
    ):
        """预测分布的均值和方差"""
        noise_pred = self.model(x, t, condition)
        x_start = self.predict_start_from_noise(x, t, noise_pred)
        x_start = torch.clamp(x_start, -1, 1)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """单步去噪"""
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, condition)
        noise = torch.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int, 
        image_size: int, 
        channels: int = 3,
        condition: Optional[torch.Tensor] = None,
        progress: bool = True
    ):
        """完整的采样过程（DDPM）"""
        shape = (batch_size, channels, image_size, image_size)
        device = next(self.model.parameters()).device
        
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        imgs = []
        
        iterator = reversed(range(0, self.timesteps))
        if progress:
            iterator = tqdm(iterator, desc="Sampling", total=self.timesteps)
        
        for i in iterator:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition)
            imgs.append(img.cpu())
        
        return img, imgs
    
    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size: int,
        image_size: int,
        channels: int = 3,
        ddim_timesteps: int = 50,
        condition: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        progress: bool = True
    ):
        """DDIM 加速采样"""
        # 选择子序列
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        
        # 预计算 DDIM 采样参数
        ddim_alpha = self.alphas_cumprod[ddim_timestep_seq]
        ddim_alpha_prev = torch.cat([torch.ones(1), self.alphas_cumprod[ddim_timestep_seq[:-1]]])
        ddim_sigma = eta * torch.sqrt(
            (1 - ddim_alpha_prev) / (1 - ddim_alpha) * (1 - ddim_alpha / ddim_alpha_prev)
        )
        
        shape = (batch_size, channels, image_size, image_size)
        device = next(self.model.parameters()).device
        
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        
        iterator = reversed(range(0, ddim_timesteps))
        if progress:
            iterator = tqdm(iterator, desc="DDIM Sampling", total=ddim_timesteps)
        
        for i in iterator:
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.model(img, t, condition)
            
            # 预测 x_0
            x_start = self.predict_start_from_noise(img, t, noise_pred)
            x_start = torch.clamp(x_start, -1, 1)
            
            if i > 0:
                alpha = ddim_alpha[i].to(device)
                alpha_prev = ddim_alpha_prev[i].to(device)
                sigma = ddim_sigma[i].to(device)
                
                # 计算 x_{t-1}
                dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
                noise = torch.randn_like(img) if eta > 0 else 0
                img = torch.sqrt(alpha_prev) * x_start + dir_xt + sigma * noise
            else:
                img = x_start
        
        return img
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        """从张量 a 中提取对应时间步的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """训练时的前向传播"""
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device, dtype=torch.long)
        return self.p_losses(x, t, condition)


class InpaintingDiffusion(GaussianDiffusion):
    """图像修复专用扩散模型"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @torch.no_grad()
    def inpaint(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
        progress: bool = True
    ):
        """
        图像修复
        Args:
            image: 原始图像 [B, C, H, W]
            mask: 修复掩码 [B, 1, H, W]，1 表示需要修复的区域
            ddim_steps: DDIM 采样步数
        """
        batch_size = image.shape[0]
        device = image.device
        
        # 生成噪声图像
        img = torch.randn_like(image)
        
        # DDIM 采样序列
        c = self.timesteps // ddim_steps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        
        iterator = reversed(range(0, ddim_steps))
        if progress:
            iterator = tqdm(iterator, desc="Inpainting", total=ddim_steps)
        
        for i in iterator:
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            
            # 在已知区域替换为加噪的原图
            known_noise = torch.randn_like(image)
            known_x_t = self.q_sample(image, t, known_noise)
            img = mask * img + (1 - mask) * known_x_t
            
            # 预测并去噪
            noise_pred = self.model(img, t)
            
            if i > 0:
                # DDIM 更新
                x_start = self.predict_start_from_noise(img, t, noise_pred)
                x_start = torch.clamp(x_start, -1, 1)
                
                # 计算下一步
                alpha = self.alphas_cumprod[ddim_timestep_seq[i]]
                alpha_prev = self.alphas_cumprod[ddim_timestep_seq[i-1]]
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
                
                dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
                noise = torch.randn_like(img) if eta > 0 else 0
                img = torch.sqrt(alpha_prev) * x_start + dir_xt + sigma * noise
            else:
                img = self.predict_start_from_noise(img, t, noise_pred)
                img = torch.clamp(img, -1, 1)
        
        # 最终合成
        result = mask * img + (1 - mask) * image
        return result


# 测试代码
if __name__ == "__main__":
    from unet import LightweightUNet
    
    # 创建模型
    model = LightweightUNet(
        in_channels=3,
        out_channels=3,
        channels=[64, 128, 256, 512]
    ).cuda()
    
    # 创建扩散模型
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=1000,
        beta_schedule="cosine",
        loss_type="l2"
    )
    
    # 测试训练损失
    x = torch.randn(4, 3, 128, 128).cuda()
    loss = diffusion(x)
    print(f"Training loss: {loss.item():.4f}")
    
    # 测试 DDIM 采样
    print("\nTesting DDIM sampling...")
    samples = diffusion.ddim_sample(
        batch_size=2,
        image_size=128,
        ddim_timesteps=50,
        progress=False
    )
    print(f"Generated samples shape: {samples.shape}")
    
    # 测试图像修复
    print("\nTesting inpainting...")
    inpaint_model = InpaintingDiffusion(
        model=model,
        timesteps=1000,
        beta_schedule="cosine"
    )
    
    image = torch.randn(2, 3, 128, 128).cuda()
    mask = torch.zeros(2, 1, 128, 128).cuda()
    mask[:, :, 32:96, 32:96] = 1  # 中心区域需要修复
    
    result = inpaint_model.inpaint(image, mask, ddim_steps=20, progress=False)
    print(f"Inpainted result shape: {result.shape}")