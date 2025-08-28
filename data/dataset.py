"""
数据集加载器
支持多种图像数据集格式和增强策略
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


class BaseImageDataset(Dataset):
    """基础图像数据集类"""
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.extensions = extensions
        
        # 收集所有图像路径
        self.image_paths = self._collect_images()
        
        # 设置默认变换
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform
            
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
    
    def _collect_images(self) -> List[Path]:
        """收集所有图像文件"""
        image_paths = []
        for ext in self.extensions:
            image_paths.extend(self.root_dir.rglob(f"*{ext}"))
            image_paths.extend(self.root_dir.rglob(f"*{ext.upper()}"))
        return sorted(image_paths)
    
    def _default_transform(self) -> transforms.Compose:
        """默认图像变换"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


class PairedImageDataset(Dataset):
    """成对图像数据集（用于图像到图像的任务）"""
    
    def __init__(
        self,
        clean_dir: str,
        degraded_dir: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None
    ):
        self.clean_dir = Path(clean_dir)
        self.degraded_dir = Path(degraded_dir)
        self.image_size = image_size
        
        # 收集成对的图像
        self.pairs = self._collect_pairs()
        
        # 设置变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        print(f"Loaded {len(self.pairs)} image pairs")
    
    def _collect_pairs(self) -> List[Tuple[Path, Path]]:
        """收集成对的图像路径"""
        pairs = []
        
        # 假设文件名相同
        for clean_path in sorted(self.clean_dir.glob("*")):
            if clean_path.is_file():
                degraded_path = self.degraded_dir / clean_path.name
                if degraded_path.exists():
                    pairs.append((clean_path, degraded_path))
                    
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_path, degraded_path = self.pairs[idx]
        
        clean_image = Image.open(clean_path).convert('RGB')
        degraded_image = Image.open(degraded_path).convert('RGB')
        
        if self.transform:
            clean_image = self.transform(clean_image)
            degraded_image = self.transform(degraded_image)
            
        return degraded_image, clean_image


class InpaintingDataset(Dataset):
    """图像修复数据集"""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        image_size: int = 256,
        random_mask: bool = True,
        mask_config: Dict[str, Any] = None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = image_size
        self.random_mask = random_mask
        
        # 默认掩码配置
        self.mask_config = mask_config or {
            'min_size': 16,
            'max_size': 128,
            'min_ratio': 0.1,
            'max_ratio': 0.4,
            'irregular_prob': 0.5
        }
        
        # 收集图像
        self.image_paths = self._collect_images()
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Loaded {len(self.image_paths)} images for inpainting")
    
    def _collect_images(self) -> List[Path]:
        """收集图像路径"""
        return sorted(self.image_dir.glob("*"))
    
    def _generate_random_mask(self, height: int, width: int) -> np.ndarray:
        """生成随机掩码"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if random.random() < self.mask_config['irregular_prob']:
            # 生成不规则掩码
            num_strokes = random.randint(1, 5)
            for _ in range(num_strokes):
                # 随机起点
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                
                # 随机笔画
                num_points = random.randint(5, 15)
                thickness = random.randint(3, 10)
                
                points = []
                for _ in range(num_points):
                    x += random.randint(-20, 20)
                    y += random.randint(-20, 20)
                    x = max(0, min(width - 1, x))
                    y = max(0, min(height - 1, y))
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.polylines(mask, [points], False, 255, thickness)
        else:
            # 生成矩形掩码
            num_rects = random.randint(1, 4)
            for _ in range(num_rects):
                w = random.randint(self.mask_config['min_size'], self.mask_config['max_size'])
                h = random.randint(self.mask_config['min_size'], self.mask_config['max_size'])
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)
                mask[y:y+h, x:x+w] = 255
                
        return mask
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # 生成或加载掩码
        if self.mask_dir and not self.random_mask:
            mask_path = self.mask_dir / f"{image_path.stem}_mask.png"
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                mask = transforms.Resize((self.image_size, self.image_size))(mask)
                mask = transforms.ToTensor()(mask)
            else:
                mask = self._generate_random_mask(self.image_size, self.image_size)
                mask = torch.from_numpy(mask).float() / 255.0
                mask = mask.unsqueeze(0)
        else:
            mask = self._generate_random_mask(self.image_size, self.image_size)
            mask = torch.from_numpy(mask).float() / 255.0
            mask = mask.unsqueeze(0)
        
        # 创建带掩码的图像
        masked_image = image * (1 - mask)
        
        return {
            'image': image,
            'mask': mask,
            'masked_image': masked_image
        }


class SuperResolutionDataset(Dataset):
    """超分辨率数据集"""
    
    def __init__(
        self,
        hr_dir: str,
        scale_factor: int = 4,
        lr_size: int = 64,
        hr_size: int = 256,
        degradation: str = 'bicubic'
    ):
        self.hr_dir = Path(hr_dir)
        self.scale_factor = scale_factor
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.degradation = degradation
        
        # 收集高分辨率图像
        self.hr_paths = sorted(self.hr_dir.glob("*"))
        
        # 高分辨率图像变换
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Loaded {len(self.hr_paths)} images for super-resolution")
    
    def _degrade_image(self, hr_image: Image.Image) -> Image.Image:
        """降级高分辨率图像"""
        lr_size = (self.lr_size, self.lr_size)
        
        if self.degradation == 'bicubic':
            lr_image = hr_image.resize(lr_size, Image.BICUBIC)
        elif self.degradation == 'bilinear':
            lr_image = hr_image.resize(lr_size, Image.BILINEAR)
        elif self.degradation == 'blur_downsample':
            # 先模糊后下采样
            hr_np = np.array(hr_image)
            blurred = cv2.GaussianBlur(hr_np, (5, 5), 1.0)
            lr_image = Image.fromarray(blurred).resize(lr_size, Image.BICUBIC)
        else:
            lr_image = hr_image.resize(lr_size, Image.BICUBIC)
            
        # 上采样回原始尺寸（用于训练）
        lr_image = lr_image.resize((self.hr_size, self.hr_size), Image.BICUBIC)
        
        return lr_image
    
    def __len__(self) -> int:
        return len(self.hr_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 加载高分辨率图像
        hr_path = self.hr_paths[idx]
        hr_image = Image.open(hr_path).convert('RGB')
        
        # 生成低分辨率图像
        lr_image = self._degrade_image(hr_image)
        
        # 应用变换
        hr_tensor = self.hr_transform(hr_image)
        lr_tensor = transforms.ToTensor()(lr_image)
        lr_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(lr_tensor)
        
        return lr_tensor, hr_tensor


class AugmentedDataset(Dataset):
    """带数据增强的数据集包装器"""
    
    def __init__(
        self,
        base_dataset: Dataset,
        augmentations: Optional[transforms.Compose] = None
    ):
        self.base_dataset = base_dataset
        
        if augmentations is None:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            ])
        else:
            self.augmentations = augmentations
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Any:
        data = self.base_dataset[idx]
        
        # 根据数据类型应用增强
        if isinstance(data, torch.Tensor):
            # 转换为 PIL 图像进行增强
            image = transforms.ToPILImage()(data * 0.5 + 0.5)
            image = self.augmentations(image)
            data = transforms.ToTensor()(image)
            data = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(data)
        elif isinstance(data, dict):
            # 对字典中的图像应用增强
            if 'image' in data:
                image = transforms.ToPILImage()(data['image'] * 0.5 + 0.5)
                image = self.augmentations(image)
                data['image'] = transforms.ToTensor()(image)
                data['image'] = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(data['image'])
                
        return data


def create_dataloader(
    dataset_type: str,
    dataset_args: dict,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """创建数据加载器"""
    
    datasets = {
        'base': BaseImageDataset,
        'paired': PairedImageDataset,
        'inpainting': InpaintingDataset,
        'super_resolution': SuperResolutionDataset,
    }
    
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataset = datasets[dataset_type](**dataset_args)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据集
    
    # 基础数据集
    print("Testing BaseImageDataset...")
    base_dataset = BaseImageDataset("./test_images", image_size=256)
    if len(base_dataset) > 0:
        sample = base_dataset[0]
        print(f"Sample shape: {sample.shape}")
    
    # 修复数据集
    print("\nTesting InpaintingDataset...")
    inpaint_dataset = InpaintingDataset("./test_images", random_mask=True)
    if len(inpaint_dataset) > 0:
        sample = inpaint_dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
    
    # 创建数据加载器
    print("\nTesting DataLoader...")
    dataloader = create_dataloader(
        'base',
        {'root_dir': './test_images', 'image_size': 256},
        batch_size=4
    )
    
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        break