"""
损失函数模块
包含各种训练损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import lpips


class DiffusionLoss(nn.Module):
    """扩散模型损失"""
    
    def __init__(
        self,
        loss_type: str = 'l2',
        lambda_vlb: float = 0.001,
        parameterization: str = 'eps'
    ):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_vlb = lambda_vlb
        self.parameterization = parameterization
        
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算损失
        Args:
            model_output: 模型预测
            target: 目标（噪声或原始图像）
            mask: 可选的掩码
        """
        if mask is not None:
            model_output = model_output * mask
            target = target * mask
            
        if self.loss_type == 'l1':
            loss = F.l1_loss(model_output, target, reduction='mean')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(model_output, target, reduction='mean')
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(model_output, target, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss


class PerceptualLoss(nn.Module):
    """感知损失（使用 LPIPS）"""
    
    def __init__(
        self,
        net_type: str = 'alex',
        device: str = 'cuda'
    ):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net_type).to(device)
        self.lpips.eval()
        
        # 冻结参数
        for param in self.lpips.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算感知损失
        Args:
            pred: 预测图像 [-1, 1]
            target: 目标图像 [-1, 1]
        """
        return self.lpips(pred, target).mean()


class StyleLoss(nn.Module):
    """风格损失（Gram 矩阵）"""
    
    def __init__(self):
        super().__init__()
        
    def gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """计算 Gram 矩阵"""
        b, c, h, w = features.shape
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算风格损失
        Args:
            pred_features: 预测图像的特征
            target_features: 目标图像的特征
        """
        pred_gram = self.gram_matrix(pred_features)
        target_gram = self.gram_matrix(target_features)
        return F.mse_loss(pred_gram, target_gram)


class TotalVariationLoss(nn.Module):
    """全变分损失（平滑性）"""
    
    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 TV 损失
        Args:
            x: 输入图像
        """
        batch_size = x.size(0)
        h = x.size(2)
        w = x.size(3)
        
        # 计算水平和垂直差分
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        
        return self.weight * (tv_h + tv_w) / (batch_size * h * w)


class AdversarialLoss(nn.Module):
    """对抗损失（用于 GAN）"""
    
    def __init__(
        self,
        loss_type: str = 'vanilla',
        real_label: float = 1.0,
        fake_label: float = 0.0
    ):
        super().__init__()
        self.loss_type = loss_type
        self.real_label = real_label
        self.fake_label = fake_label
        
        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'hinge':
            self.criterion = None
        else:
            raise ValueError(f"Unknown adversarial loss type: {loss_type}")
            
    def forward(
        self,
        pred: torch.Tensor,
        is_real: bool
    ) -> torch.Tensor:
        """
        计算对抗损失
        Args:
            pred: 判别器输出
            is_real: 是否为真实样本
        """
        if self.loss_type == 'hinge':
            if is_real:
                loss = F.relu(1 - pred).mean()
            else:
                loss = F.relu(1 + pred).mean()
        else:
            target = torch.ones_like(pred) * (self.real_label if is_real else self.fake_label)
            loss = self.criterion(pred, target)
            
        return loss


class CombinedLoss(nn.Module):
    """组合损失"""
    
    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Dict[str, float]
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        Args:
            pred: 预测
            target: 目标
            **kwargs: 额外参数
        """
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            if name in self.weights:
                loss_val = loss_fn(pred, target, **kwargs)
                weighted_loss = self.weights[name] * loss_val
                total_loss += weighted_loss
                loss_dict[name] = loss_val
                
        loss_dict['total'] = total_loss
        return loss_dict


class FocalFrequencyLoss(nn.Module):
    """焦点频率损失（用于保留高频细节）"""
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算频域损失
        """
        # 转换到频域
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        
        # 计算幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 计算权重（高频部分权重更大）
        b, c, h, w = pred_mag.shape
        freq_weight = self._get_frequency_weight(h, w, pred_mag.device)
        
        # 加权损失
        loss = F.l1_loss(pred_mag * freq_weight, target_mag * freq_weight)
        
        return self.loss_weight * loss
    
    def _get_frequency_weight(
        self,
        h: int,
        w: int,
        device: torch.device
    ) -> torch.Tensor:
        """生成频率权重"""
        # 创建频率坐标
        freq_h = torch.fft.fftfreq(h, device=device).view(-1, 1)
        freq_w = torch.fft.rfftfreq(w, device=device).view(1, -1)
        
        # 计算频率半径
        freq_radius = torch.sqrt(freq_h ** 2 + freq_w ** 2)
        
        # 生成权重（高频权重更大）
        weight = 1 + self.alpha * freq_radius
        
        return weight.unsqueeze(0).unsqueeze(0)


class ContrastiveLoss(nn.Module):
    """对比学习损失"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        similarity: str = 'cosine'
    ):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
        
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算对比损失
        Args:
            features1: 第一组特征
            features2: 第二组特征
            labels: 可选的标签
        """
        # 归一化特征
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # 计算相似度矩阵
        if self.similarity == 'cosine':
            sim_matrix = torch.matmul(features1, features2.T) / self.temperature
        else:
            # 欧氏距离
            sim_matrix = -torch.cdist(features1, features2) / self.temperature
            
        # 创建标签（如果未提供）
        if labels is None:
            batch_size = features1.shape[0]
            labels = torch.arange(batch_size, device=features1.device)
            
        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier 损失（鲁棒的 L1 损失）"""
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Charbonnier 损失
        """
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class EdgeLoss(nn.Module):
    """边缘损失（保留边缘信息）"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
        # Sobel 算子
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算边缘损失
        """
        # 转换为灰度
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        
        # 移动 Sobel 算子到正确的设备
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        
        # 计算梯度
        pred_edge_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
        
        target_edge_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)
        
        # 计算损失
        loss = F.l1_loss(pred_edge, target_edge)
        
        return self.weight * loss


def create_loss(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """创建损失函数"""
    losses = {
        'diffusion': DiffusionLoss,
        'perceptual': PerceptualLoss,
        'style': StyleLoss,
        'tv': TotalVariationLoss,
        'adversarial': AdversarialLoss,
        'focal_frequency': FocalFrequencyLoss,
        'contrastive': ContrastiveLoss,
        'charbonnier': CharbonnierLoss,
        'edge': EdgeLoss,
    }
    
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}")
        
    return losses[loss_type](**kwargs)


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    pred = torch.randn(4, 3, 256, 256).to(device)
    target = torch.randn(4, 3, 256, 256).to(device)
    
    # 测试各种损失
    print("Testing losses:")
    
    # 扩散损失
    diff_loss = DiffusionLoss()
    loss = diff_loss(pred, target)
    print(f"Diffusion loss: {loss.item():.4f}")
    
    # TV 损失
    tv_loss = TotalVariationLoss()
    loss = tv_loss(pred)
    print(f"TV loss: {loss.item():.4f}")
    
    # Charbonnier 损失
    char_loss = CharbonnierLoss()
    loss = char_loss(pred, target)
    print(f"Charbonnier loss: {loss.item():.4f}")
    
    # 边缘损失
    edge_loss = EdgeLoss()
    loss = edge_loss(pred, target)
    print(f"Edge loss: {loss.item():.4f}")
    
    # 组合损失
    combined = CombinedLoss(
        losses={
            'l2': nn.MSELoss(),
            'tv': TotalVariationLoss(),
            'edge': EdgeLoss()
        },
        weights={
            'l2': 1.0,
            'tv': 0.01,
            'edge': 0.1
        }
    )
    
    loss_dict = combined(pred, target)
    print(f"Combined losses: {loss_dict}")