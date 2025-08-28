"""
优化器配置模块
提供各种优化器和学习率调度器
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, Dict, Any, List
import numpy as np


class WarmupLR(_LRScheduler):
    """带预热的学习率调度器"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 线性预热
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        return self.base_lrs


class CosineAnnealingWarmupLR(_LRScheduler):
    """余弦退火 + 预热"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 预热阶段
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """多项式学习率调度"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_steps: int,
        power: float = 0.9,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.max_steps = max_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        progress = min(self.last_epoch / self.max_steps, 1.0)
        factor = (1 - progress) ** self.power
        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


class ExponentialWarmupLR(_LRScheduler):
    """指数预热学习率"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        gamma: float = 0.95,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 预热阶段
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # 指数衰减
            return [
                base_lr * (self.gamma ** (self.last_epoch - self.warmup_steps))
                for base_lr in self.base_lrs
            ]


class CyclicLR(_LRScheduler):
    """循环学习率"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: str = 'triangular',
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** self.last_epoch
        else:
            scale_factor = 1.0
            
        base_height = (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        
        return [self.base_lr + base_height for _ in self.base_lrs]


class AdaptiveLR(_LRScheduler):
    """自适应学习率（基于损失）"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        threshold: float = 1e-4,
        last_epoch: int = -1
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        self.best = float('inf')
        self.num_bad_epochs = 0
        super().__init__(optimizer, last_epoch)
        
    def step(self, metric: float):
        """
        更新学习率
        Args:
            metric: 当前指标（如验证损失）
        """
        if metric < self.best - self.threshold:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
            self.num_bad_epochs = 0
            
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    lr: float,
    **kwargs
) -> optim.Optimizer:
    """
    创建优化器
    Args:
        model: 模型
        optimizer_type: 优化器类型
        lr: 学习率
        **kwargs: 其他参数
    """
    params = model.parameters()
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            params,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0),
            nesterov=kwargs.get('nesterov', False)
        )
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            params,
            lr=lr,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    elif optimizer_type.lower() == 'adagrad':
        optimizer = optim.Adagrad(
            params,
            lr=lr,
            lr_decay=kwargs.get('lr_decay', 0),
            weight_decay=kwargs.get('weight_decay', 0),
            eps=kwargs.get('eps', 1e-10)
        )
    elif optimizer_type.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            params,
            lr=lr,
            rho=kwargs.get('rho', 0.9),
            eps=kwargs.get('eps', 1e-6),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    elif optimizer_type.lower() == 'radam':
        # RAdam 优化器
        optimizer = RAdam(
            params,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    elif optimizer_type.lower() == 'lamb':
        # LAMB 优化器（适合大批量训练）
        optimizer = LAMB(
            params,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-6),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    **kwargs
) -> _LRScheduler:
    """
    创建学习率调度器
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        **kwargs: 其他参数
    """
    if scheduler_type.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type.lower() == 'cosine_warmup':
        scheduler = CosineAnnealingWarmupLR(
            optimizer,
            warmup_steps=kwargs.get('warmup_steps', 1000),
            max_steps=kwargs.get('max_steps', 10000),
            min_lr=kwargs.get('min_lr', 0)
        )
    elif scheduler_type.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [30, 60, 90]),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type.lower() == 'polynomial':
        scheduler = PolynomialLR(
            optimizer,
            max_steps=kwargs.get('max_steps', 10000),
            power=kwargs.get('power', 0.9),
            min_lr=kwargs.get('min_lr', 0)
        )
    elif scheduler_type.lower() == 'warmup':
        scheduler = WarmupLR(
            optimizer,
            warmup_steps=kwargs.get('warmup_steps', 1000)
        )
    elif scheduler_type.lower() == 'cyclic':
        scheduler = CyclicLR(
            optimizer,
            base_lr=kwargs.get('base_lr', 1e-4),
            max_lr=kwargs.get('max_lr', 1e-3),
            step_size=kwargs.get('step_size', 2000),
            mode=kwargs.get('mode', 'triangular')
        )
    elif scheduler_type.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type.lower() == 'adaptive':
        scheduler = AdaptiveLR(
            optimizer,
            patience=kwargs.get('patience', 10),
            factor=kwargs.get('factor', 0.5),
            min_lr=kwargs.get('min_lr', 1e-6),
            threshold=kwargs.get('threshold', 1e-4)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    return scheduler


# 自定义优化器实现
class RAdam(optim.Optimizer):
    """Rectified Adam 优化器"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                    
                p_data_fp32 = p.data.float()
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                            (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                    
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    
                p.data.copy_(p_data_fp32)
                
        return loss


class LAMB(optim.Optimizer):
    """Layer-wise Adaptive Moments optimizer for Batch training"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                    
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 指数移动平均
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 自适应学习率
                adam_step = exp_avg / bias_correction1 / (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + group['eps']
                )
                
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])
                    
                # Layer adaptation
                param_norm = p.data.norm()
                adam_norm = adam_step.norm()
                
                if param_norm > 0 and adam_norm > 0:
                    trust_ratio = param_norm / adam_norm
                else:
                    trust_ratio = 1
                    
                p.data.add_(adam_step, alpha=-group['lr'] * trust_ratio)
                
        return loss


if __name__ == "__main__":
    # 测试优化器和调度器
    import matplotlib.pyplot as plt
    
    # 创建简单模型
    model = torch.nn.Linear(10, 1)
    
    # 测试不同优化器
    optimizers = ['adam', 'adamw', 'sgd', 'radam']
    for opt_type in optimizers:
        print(f"Testing {opt_type} optimizer...")
        optimizer = create_optimizer(model, opt_type, lr=0.001)
        print(f"  Created: {optimizer.__class__.__name__}")
        
    # 测试不同调度器
    optimizer = create_optimizer(model, 'adam', lr=0.01)
    
    schedulers = ['cosine', 'cosine_warmup', 'step', 'polynomial']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, sched_type in enumerate(schedulers):
        print(f"Testing {sched_type} scheduler...")
        
        # 重新创建优化器
        optimizer = create_optimizer(model, 'adam', lr=0.01)
        
        scheduler = create_scheduler(
            optimizer,
            sched_type,
            T_max=100,
            warmup_steps=10,
            max_steps=100,
            step_size=30,
            gamma=0.1
        )
        
        lrs = []
        for epoch in range(100):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
            
        axes[i].plot(lrs)
        axes[i].set_title(f'{sched_type} Scheduler')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Learning Rate')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('scheduler_comparison.png')
    print("Scheduler comparison saved to scheduler_comparison.png")