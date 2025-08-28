"""
评估指标模块
提供全面的图像质量评估指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Dict, List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """综合指标计算器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lpips_fn = None  # 延迟加载
        
    def _ensure_lpips(self):
        """确保 LPIPS 已加载"""
        if self.lpips_fn is None:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_fn.eval()
            except ImportError:
                print("Warning: lpips not installed, LPIPS metric will not be available")
    
    @staticmethod
    def calculate_psnr(
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        max_val: float = 1.0
    ) -> float:
        """
        计算 PSNR (Peak Signal-to-Noise Ratio)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
            
        # 确保形状匹配
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
            
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
            
        return 20 * np.log10(max_val / np.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        multichannel: bool = True,
        data_range: float = 1.0
    ) -> float:
        """
        计算 SSIM (Structural Similarity Index)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
            
        # 处理维度
        if pred.ndim == 4:  # batch dimension
            pred = pred[0]
            target = target[0]
        if pred.ndim == 3 and pred.shape[0] in [1, 3]:  # CHW -> HWC
            pred = np.transpose(pred, (1, 2, 0))
            target = np.transpose(target, (1, 2, 0))
            
        return ssim(target, pred, multichannel=multichannel, data_range=data_range, channel_axis=2)
    
    def calculate_lpips(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算 LPIPS (Learned Perceptual Image Patch Similarity)
        """
        self._ensure_lpips()
        if self.lpips_fn is None:
            return -1.0
            
        # 转换为张量
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
            
        # 调整维度
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)
            
        # 确保在正确的设备上
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # 归一化到 [-1, 1]
        if pred.max() > 1:
            pred = pred / 127.5 - 1
            target = target / 127.5 - 1
            
        with torch.no_grad():
            score = self.lpips_fn(pred, target)
            
        return score.item()
    
    @staticmethod
    def calculate_mae(
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算 MAE (Mean Absolute Error)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
            
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def calculate_mse(
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算 MSE (Mean Squared Error)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
            
        return np.mean((pred - target) ** 2)
    
    def calculate_fid(
        self,
        real_features: np.ndarray,
        fake_features: np.ndarray
    ) -> float:
        """
        计算 FID (Fréchet Inception Distance)
        简化版本，需要预先提取的特征
        """
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # 计算 Fréchet 距离
        diff = mu_real - mu_fake
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        
        # 处理数值误差
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return float(fid)
    
    @staticmethod
    def calculate_is(
        probs: np.ndarray,
        splits: int = 10
    ) -> Tuple[float, float]:
        """
        计算 IS (Inception Score)
        Args:
            probs: 预测的概率分布 [N, num_classes]
            splits: 分割数量
        """
        N = probs.shape[0]
        split_size = N // splits
        
        scores = []
        for i in range(splits):
            part = probs[i * split_size:(i + 1) * split_size]
            
            # 计算 KL 散度
            py = np.mean(part, axis=0)
            kl_div = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
            kl_div = np.mean(np.sum(kl_div, axis=1))
            
            scores.append(np.exp(kl_div))
            
        return float(np.mean(scores)), float(np.std(scores))
    
    @staticmethod
    def calculate_niqe(image: np.ndarray) -> float:
        """
        计算 NIQE (Natural Image Quality Evaluator)
        无参考图像质量评估
        """
        # 简化版本的 NIQE
        # 实际实现需要预训练的自然场景统计模型
        
        # 转换为灰度图
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 计算局部均值和方差
        kernel = cv2.getGaussianKernel(7, 1.5)
        kernel = np.outer(kernel, kernel)
        
        mu = cv2.filter2D(gray, -1, kernel)
        mu_sq = cv2.filter2D(gray * gray, -1, kernel)
        sigma = np.sqrt(np.maximum(mu_sq - mu * mu, 0))
        
        # 计算特征统计
        features = []
        features.append(np.mean(mu))
        features.append(np.std(mu))
        features.append(np.mean(sigma))
        features.append(np.std(sigma))
        
        # 简化的质量分数（实际需要与自然图像模型比较）
        score = np.mean(features)
        
        return float(score)
    
    def calculate_all(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        calculate_expensive: bool = False
    ) -> Dict[str, float]:
        """
        计算所有指标
        Args:
            pred: 预测图像
            target: 目标图像
            calculate_expensive: 是否计算昂贵的指标（如 LPIPS）
        """
        metrics = {
            'psnr': self.calculate_psnr(pred, target),
            'ssim': self.calculate_ssim(pred, target),
            'mae': self.calculate_mae(pred, target),
            'mse': self.calculate_mse(pred, target),
        }
        
        if calculate_expensive:
            metrics['lpips'] = self.calculate_lpips(pred, target)
            
        return metrics
    
    def calculate_batch(
        self,
        preds: Union[List, torch.Tensor],
        targets: Union[List, torch.Tensor],
        metrics_list: List[str] = ['psnr', 'ssim']
    ) -> Dict[str, float]:
        """
        批量计算指标
        """
        results = {metric: [] for metric in metrics_list}
        
        # 确保是列表格式
        if isinstance(preds, torch.Tensor):
            preds = [preds[i] for i in range(preds.shape[0])]
        if isinstance(targets, torch.Tensor):
            targets = [targets[i] for i in range(targets.shape[0])]
            
        for pred, target in zip(preds, targets):
            for metric in metrics_list:
                if metric == 'psnr':
                    value = self.calculate_psnr(pred, target)
                elif metric == 'ssim':
                    value = self.calculate_ssim(pred, target)
                elif metric == 'mae':
                    value = self.calculate_mae(pred, target)
                elif metric == 'mse':
                    value = self.calculate_mse(pred, target)
                elif metric == 'lpips':
                    value = self.calculate_lpips(pred, target)
                else:
                    continue
                    
                results[metric].append(value)
                
        # 计算平均值
        return {metric: np.mean(values) for metric, values in results.items()}


class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self, metrics_names: List[str]):
        self.metrics_names = metrics_names
        self.history = {name: [] for name in metrics_names}
        self.best = {name: None for name in metrics_names}
        self.current = {name: 0 for name in metrics_names}
        
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for name, value in metrics.items():
            if name in self.metrics_names:
                self.history[name].append(value)
                self.current[name] = value
                
                # 更新最佳值
                if self.best[name] is None:
                    self.best[name] = value
                else:
                    # PSNR, SSIM 越大越好，其他越小越好
                    if name in ['psnr', 'ssim', 'is']:
                        if value > self.best[name]:
                            self.best[name] = value
                    else:
                        if value < self.best[name]:
                            self.best[name] = value
                            
    def get_current(self) -> Dict[str, float]:
        """获取当前值"""
        return self.current.copy()
    
    def get_best(self) -> Dict[str, float]:
        """获取最佳值"""
        return self.best.copy()
    
    def get_history(self, metric_name: str) -> List[float]:
        """获取历史记录"""
        return self.history.get(metric_name, [])
    
    def is_best(self, metric_name: str) -> bool:
        """检查当前是否为最佳"""
        if metric_name not in self.metrics_names:
            return False
        return self.current[metric_name] == self.best[metric_name]
    
    def save(self, path: str):
        """保存历史记录"""
        import json
        data = {
            'history': self.history,
            'best': self.best,
            'current': self.current
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, path: str):
        """加载历史记录"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.history = data['history']
        self.best = data['best']
        self.current = data['current']


def benchmark_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metrics_calculator: MetricsCalculator,
    device: torch.device
) -> Dict[str, float]:
    """
    在数据集上评估模型
    """
    model.eval()
    tracker = MetricTracker(['psnr', 'ssim', 'mae'])
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                inputs, targets = batch
            else:
                inputs = batch
                targets = batch  # 自编码器场景
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 模型推理
            outputs = model(inputs)
            
            # 计算指标
            metrics = metrics_calculator.calculate_batch(
                outputs, targets, 
                metrics_list=['psnr', 'ssim', 'mae']
            )
            tracker.update(metrics)
            
    return tracker.get_current()


if __name__ == "__main__":
    # 测试指标计算
    print("Testing metrics calculator...")
    
    calculator = MetricsCalculator()
    
    # 创建测试图像
    pred = np.random.rand(256, 256, 3).astype(np.float32)
    target = np.random.rand(256, 256, 3).astype(np.float32)
    
    # 计算单个指标
    psnr_val = calculator.calculate_psnr(pred, target)
    ssim_val = calculator.calculate_ssim(pred, target)
    mae_val = calculator.calculate_mae(pred, target)
    
    print(f"PSNR: {psnr_val:.2f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"MAE: {mae_val:.4f}")
    
    # 计算所有指标
    all_metrics = calculator.calculate_all(pred, target)
    print(f"All metrics: {all_metrics}")
    
    # 测试指标跟踪器
    print("\nTesting metric tracker...")
    tracker = MetricTracker(['psnr', 'ssim', 'lpips'])
    
    for i in range(5):
        metrics = {
            'psnr': np.random.uniform(20, 30),
            'ssim': np.random.uniform(0.7, 0.9),
            'lpips': np.random.uniform(0.1, 0.3)
        }
        tracker.update(metrics)
        
    print(f"Current: {tracker.get_current()}")
    print(f"Best: {tracker.get_best()}")
    
    # 保存和加载测试
    tracker.save('test_metrics.json')
    print("Metrics saved successfully!")