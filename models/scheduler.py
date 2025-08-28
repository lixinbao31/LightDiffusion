"""
采样调度器
实现各种噪声调度策略（DDPM, DDIM, DPM-Solver等）
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod


class NoiseScheduler(ABC):
    """噪声调度器基类"""
    
    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """设置推理时间步"""
        pass
    
    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """执行一步去噪"""
        pass


class DDPMScheduler(NoiseScheduler):
    """DDPM (Denoising Diffusion Probabilistic Models) 调度器"""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        
        # 初始化 beta 值
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # 计算用于采样的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验分布参数
        self.posterior_variance = self._get_variance()
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def _get_beta_schedule(self) -> torch.Tensor:
        """获取 beta 调度"""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "scaled_linear":
            return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps) ** 2
        elif self.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """余弦调度"""
        steps = self.num_train_timesteps + 1
        s = 0.008
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def _get_variance(self) -> torch.Tensor:
        """获取方差"""
        if self.variance_type == "fixed_small":
            variance = self.posterior_variance
        elif self.variance_type == "fixed_large":
            variance = self.betas
        else:
            raise ValueError(f"Unknown variance type: {self.variance_type}")
        return variance
    
    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """设置推理步数"""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.timesteps = torch.arange(0, self.num_train_timesteps, step_ratio).flip(0).to(device)
        
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """向干净样本添加噪声"""
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps].flatten()
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
            
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """执行一步 DDPM 采样"""
        t = timestep
        
        # 预测原始样本
        pred_original_sample = (
            sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
        ) / self.sqrt_alphas_cumprod[t]
        
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 计算后验均值
        pred_prev_sample = (
            self.posterior_mean_coef1[t] * pred_original_sample +
            self.posterior_mean_coef2[t] * sample
        )
        
        # 添加噪声
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output, generator=generator)
            variance = torch.sqrt(self.posterior_variance[t]) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample


class DDIMScheduler(NoiseScheduler):
    """DDIM (Denoising Diffusion Implicit Models) 调度器"""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        eta: float = 0.0,
        clip_sample: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.eta = eta
        self.clip_sample = clip_sample
        
        # 初始化参数（与 DDPM 相同）
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.final_alpha_cumprod = self.alphas_cumprod[-1]
        
    def _get_beta_schedule(self) -> torch.Tensor:
        """获取 beta 调度"""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "cosine":
            steps = self.num_train_timesteps + 1
            s = 0.008
            x = torch.linspace(0, self.num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """设置 DDIM 推理步数"""
        self.num_inference_steps = num_inference_steps
        
        # DDIM 采样的时间步选择
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.timesteps = torch.arange(0, self.num_train_timesteps, step_ratio).flip(0).long()
        self.timesteps = self.timesteps.to(device)
        
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """执行一步 DDIM 采样"""
        if eta is None:
            eta = self.eta
            
        # 获取当前和前一个时间步的 alpha 值
        alpha_prod_t = self.alphas_cumprod[timestep]
        
        # 获取前一个时间步
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        # 计算 beta
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 预测原始样本
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 计算方差
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5
        
        # 计算方向指向 x_t
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output
        
        # 计算前一个样本
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # 添加噪声
        if eta > 0 and timestep > 0:
            noise = torch.randn_like(model_output, generator=generator)
            prev_sample = prev_sample + std_dev_t * noise
            
        return prev_sample


class DPMSolverScheduler(NoiseScheduler):
    """DPM-Solver 调度器（快速采样）"""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        solver_order: int = 2,
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        
        # 初始化噪声调度
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # DPM-Solver 特定参数
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        
        # 存储历史预测用于高阶求解
        self.model_outputs = []
        
    def _get_beta_schedule(self) -> torch.Tensor:
        """获取 beta 调度"""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "cosine":
            steps = self.num_train_timesteps + 1
            s = 0.008
            x = torch.linspace(0, self.num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """设置推理步数"""
        self.num_inference_steps = num_inference_steps
        
        # DPM-Solver 使用对数空间均匀采样
        t = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1).round()
        self.timesteps = t.flip(0)[:-1].long().to(device)
        
        # 清空历史
        self.model_outputs = []
        
    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """一阶 DPM-Solver 更新"""
        lambda_t = self.lambda_t[timestep]
        lambda_s = self.lambda_t[prev_timestep]
        alpha_t = self.alpha_t[timestep]
        alpha_s = self.alpha_t[prev_timestep]
        sigma_t = self.sigma_t[timestep]
        
        h = lambda_s - lambda_t
        
        if self.prediction_type == "epsilon":
            x_0 = (sample - sigma_t * model_output) / alpha_t
        else:
            x_0 = model_output
            
        x_t = alpha_s * x_0 + sigma_t * torch.exp(-h) * model_output
        
        return x_t
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """执行一步 DPM-Solver 采样"""
        # 获取前一个时间步
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_timestep = self.timesteps[step_index + 1] if step_index < len(self.timesteps) - 1 else 0
        
        # 使用一阶更新（可扩展到高阶）
        prev_sample = self.dpm_solver_first_order_update(
            model_output, timestep, prev_timestep, sample
        )
        
        return prev_sample


# 辅助函数
def create_scheduler(scheduler_type: str, **kwargs) -> NoiseScheduler:
    """创建调度器"""
    schedulers = {
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "dpm": DPMSolverScheduler,
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return schedulers[scheduler_type](**kwargs)


if __name__ == "__main__":
    # 测试调度器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试 DDPM
    ddpm = DDPMScheduler()
    ddpm.set_timesteps(50, device)
    print(f"DDPM timesteps: {ddpm.timesteps[:5]}...")
    
    # 测试 DDIM
    ddim = DDIMScheduler(eta=0.0)
    ddim.set_timesteps(20, device)
    print(f"DDIM timesteps: {ddim.timesteps[:5]}...")
    
    # 测试 DPM-Solver
    dpm = DPMSolverScheduler()
    dpm.set_timesteps(10, device)
    print(f"DPM timesteps: {dpm.timesteps[:5]}...")
    
    # 测试采样步骤
    sample = torch.randn(1, 3, 64, 64).to(device)
    noise_pred = torch.randn_like(sample)
    
    for scheduler in [ddpm, ddim, dpm]:
        result = scheduler.step(noise_pred, scheduler.timesteps[0], sample)
        print(f"{scheduler.__class__.__name__} output shape: {result.shape}")