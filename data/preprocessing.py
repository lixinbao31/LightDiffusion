"""
图像预处理模块
提供各种图像预处理和后处理功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, List, Optional
from torchvision import transforms


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        device: str = 'cuda'
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def preprocess_image(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor, str],
        return_type: str = 'tensor'
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        预处理单张图像
        Args:
            image: 输入图像（numpy数组、PIL图像、张量或路径）
            return_type: 返回类型 ('tensor' 或 'numpy')
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:  # 灰度图
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = transforms.ToPILImage()(image)
        
        # 调整大小
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # 转换为张量
        tensor = transforms.ToTensor()(image)
        
        # 归一化到 [-1, 1]
        if self.normalize:
            tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(tensor)
        
        # 添加批次维度
        tensor = tensor.unsqueeze(0).to(self.device)
        
        if return_type == 'numpy':
            return self.tensor_to_numpy(tensor)
        return tensor
    
    def preprocess_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        return_type: str = 'tensor'
    ) -> Union[torch.Tensor, np.ndarray]:
        """预处理批量图像"""
        tensors = []
        for image in images:
            tensor = self.preprocess_image(image, return_type='tensor')
            tensors.append(tensor)
        
        batch = torch.cat(tensors, dim=0)
        
        if return_type == 'numpy':
            return self.tensor_to_numpy(batch)
        return batch
    
    def postprocess_image(
        self,
        tensor: torch.Tensor,
        denormalize: bool = True,
        return_type: str = 'numpy'
    ) -> Union[np.ndarray, Image.Image]:
        """
        后处理模型输出
        Args:
            tensor: 模型输出张量
            denormalize: 是否反归一化
            return_type: 返回类型 ('numpy', 'pil')
        """
        # 移除批次维度
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # 反归一化
        if denormalize and self.normalize:
            tensor = tensor * 0.5 + 0.5
        
        # 限制范围
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为 numpy
        image = tensor.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        
        if return_type == 'pil':
            return Image.fromarray(image)
        return image
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """张量转 numpy"""
        return tensor.detach().cpu().numpy()
    
    @staticmethod
    def numpy_to_tensor(array: np.ndarray, device: str = 'cuda') -> torch.Tensor:
        """numpy 转张量"""
        tensor = torch.from_numpy(array).float()
        if array.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        return tensor.to(device)


class NoiseGenerator:
    """噪声生成器"""
    
    @staticmethod
    def gaussian_noise(
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """生成高斯噪声"""
        noise = torch.randn(shape, device=device) * std + mean
        return noise
    
    @staticmethod
    def salt_pepper_noise(
        image: torch.Tensor,
        prob: float = 0.05
    ) -> torch.Tensor:
        """添加椒盐噪声"""
        noise = torch.rand_like(image)
        image = image.clone()
        image[noise < prob/2] = -1  # 盐噪声
        image[noise > 1 - prob/2] = 1  # 椒噪声
        return image
    
    @staticmethod
    def poisson_noise(image: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """添加泊松噪声"""
        # 转换到正值范围
        image_pos = (image + 1) / 2
        
        # 应用泊松噪声
        noisy = torch.poisson(image_pos * 255 * scale) / (255 * scale)
        
        # 转换回 [-1, 1]
        noisy = noisy * 2 - 1
        return torch.clamp(noisy, -1, 1)


class MaskGenerator:
    """掩码生成器"""
    
    @staticmethod
    def rectangular_mask(
        height: int,
        width: int,
        rect_size: Optional[Tuple[int, int]] = None,
        rect_pos: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """生成矩形掩码"""
        mask = torch.zeros(1, 1, height, width)
        
        if rect_size is None:
            rect_h = np.random.randint(height // 4, height // 2)
            rect_w = np.random.randint(width // 4, width // 2)
        else:
            rect_h, rect_w = rect_size
            
        if rect_pos is None:
            y = np.random.randint(0, height - rect_h)
            x = np.random.randint(0, width - rect_w)
        else:
            y, x = rect_pos
            
        mask[:, :, y:y+rect_h, x:x+rect_w] = 1
        return mask
    
    @staticmethod
    def irregular_mask(
        height: int,
        width: int,
        min_strokes: int = 1,
        max_strokes: int = 5,
        min_width: int = 10,
        max_width: int = 40
    ) -> torch.Tensor:
        """生成不规则掩码"""
        mask = np.zeros((height, width), dtype=np.float32)
        
        num_strokes = np.random.randint(min_strokes, max_strokes + 1)
        
        for _ in range(num_strokes):
            # 随机起点
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            # 随机笔画宽度
            w = np.random.randint(min_width, max_width + 1)
            
            # 随机笔画长度和方向
            num_points = np.random.randint(5, 15)
            points = [[x, y]]
            
            for _ in range(num_points):
                angle = np.random.uniform(0, 2 * np.pi)
                step = np.random.uniform(10, 30)
                x = int(x + step * np.cos(angle))
                y = int(y + step * np.sin(angle))
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                points.append([x, y])
            
            # 绘制笔画
            points = np.array(points, dtype=np.int32)
            cv2.polylines(mask, [points], False, 1.0, thickness=w)
            
        # 转换为张量
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        return mask
    
    @staticmethod
    def center_mask(
        height: int,
        width: int,
        mask_ratio: float = 0.25
    ) -> torch.Tensor:
        """生成中心掩码"""
        mask = torch.zeros(1, 1, height, width)
        
        h = int(height * np.sqrt(mask_ratio))
        w = int(width * np.sqrt(mask_ratio))
        
        y = (height - h) // 2
        x = (width - w) // 2
        
        mask[:, :, y:y+h, x:x+w] = 1
        return mask


class ImageAugmentation:
    """图像增强"""
    
    @staticmethod
    def random_crop(
        image: torch.Tensor,
        crop_size: Tuple[int, int]
    ) -> torch.Tensor:
        """随机裁剪"""
        _, _, h, w = image.shape
        crop_h, crop_w = crop_size
        
        if h < crop_h or w < crop_w:
            # 如果图像小于裁剪尺寸，先上采样
            image = F.interpolate(image, size=(crop_h, crop_w), mode='bilinear')
            return image
            
        y = np.random.randint(0, h - crop_h + 1)
        x = np.random.randint(0, w - crop_w + 1)
        
        return image[:, :, y:y+crop_h, x:x+crop_w]
    
    @staticmethod
    def center_crop(
        image: torch.Tensor,
        crop_size: Tuple[int, int]
    ) -> torch.Tensor:
        """中心裁剪"""
        _, _, h, w = image.shape
        crop_h, crop_w = crop_size
        
        y = (h - crop_h) // 2
        x = (w - crop_w) // 2
        
        return image[:, :, y:y+crop_h, x:x+crop_w]
    
    @staticmethod
    def random_flip(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """随机翻转"""
        if torch.rand(1) < p:
            image = torch.flip(image, dims=[-1])  # 水平翻转
        if torch.rand(1) < p:
            image = torch.flip(image, dims=[-2])  # 垂直翻转
        return image
    
    @staticmethod
    def random_rotation(
        image: torch.Tensor,
        max_angle: float = 30
    ) -> torch.Tensor:
        """随机旋转"""
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = angle * np.pi / 180
        
        # 创建旋转矩阵
        theta = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0]
        ], dtype=torch.float).unsqueeze(0)
        
        # 创建网格
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        
        # 应用旋转
        rotated = F.grid_sample(image, grid, align_corners=False)
        
        return rotated
    
    @staticmethod
    def color_jitter(
        image: torch.Tensor,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ) -> torch.Tensor:
        """颜色抖动"""
        # 亮度
        if brightness > 0:
            factor = torch.empty(1).uniform_(1 - brightness, 1 + brightness)
            image = image * factor
            
        # 对比度
        if contrast > 0:
            factor = torch.empty(1).uniform_(1 - contrast, 1 + contrast)
            mean = image.mean(dim=[2, 3], keepdim=True)
            image = (image - mean) * factor + mean
            
        return torch.clamp(image, -1, 1)


def create_degradation(
    image: torch.Tensor,
    degradation_type: str,
    **kwargs
) -> torch.Tensor:
    """
    创建退化图像
    Args:
        image: 输入图像
        degradation_type: 退化类型 ('noise', 'blur', 'downsample', 'jpeg')
    """
    if degradation_type == 'noise':
        noise_level = kwargs.get('noise_level', 0.1)
        noise = torch.randn_like(image) * noise_level
        degraded = image + noise
        
    elif degradation_type == 'blur':
        kernel_size = kwargs.get('kernel_size', 5)
        sigma = kwargs.get('sigma', 1.0)
        
        # 创建高斯核
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.outer(kernel, kernel)
        kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)
        
        # 应用模糊
        degraded = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
        
    elif degradation_type == 'downsample':
        scale = kwargs.get('scale', 4)
        _, _, h, w = image.shape
        
        # 下采样
        degraded = F.interpolate(image, size=(h//scale, w//scale), mode='bilinear')
        # 上采样回原始尺寸
        degraded = F.interpolate(degraded, size=(h, w), mode='bilinear')
        
    elif degradation_type == 'jpeg':
        quality = kwargs.get('quality', 30)
        # 这里简化处理，实际 JPEG 压缩更复杂
        # 添加块效应和颜色失真
        degraded = image
        # 模拟 JPEG 块效应
        block_size = 8
        _, _, h, w = image.shape
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = image[:, :, i:i+block_size, j:j+block_size]
                mean = block.mean(dim=[2, 3], keepdim=True)
                degraded[:, :, i:i+block_size, j:j+block_size] = mean
                
    else:
        degraded = image
        
    return torch.clamp(degraded, -1, 1)


if __name__ == "__main__":
    # 测试预处理器
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 测试预处理
    tensor = preprocessor.preprocess_image(test_image)
    print(f"Preprocessed tensor shape: {tensor.shape}")
    
    # 测试后处理
    output = preprocessor.postprocess_image(tensor)
    print(f"Postprocessed image shape: {output.shape}")
    
    # 测试噪声生成
    noise = NoiseGenerator.gaussian_noise((1, 3, 256, 256))
    print(f"Noise shape: {noise.shape}")
    
    # 测试掩码生成
    mask = MaskGenerator.irregular_mask(256, 256)
    print(f"Mask shape: {mask.shape}")
    
    # 测试图像增强
    augmented = ImageAugmentation.random_crop(tensor, (128, 128))
    print(f"Augmented shape: {augmented.shape}")