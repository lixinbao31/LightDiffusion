"""
训练脚本 - train.py
支持分布式训练、混合精度训练、断点续训等功能
"""

import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from PIL import Image

from models.unet import LightweightUNet
from models.diffusion import GaussianDiffusion


class ImageDataset(Dataset):
    """图像数据集"""
    
    def __init__(self, root_dir, transform=None, image_size=256):
        self.root_dir = Path(root_dir)
        self.transform = transform or self.default_transform(image_size)
        
        # 收集所有图像文件
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
            self.image_paths.extend(self.root_dir.rglob(ext))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
    def default_transform(self, image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)


class Trainer:
    """训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir']) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # 初始化模型
        self.model = self._build_model()
        self.diffusion = GaussianDiffusion(
            model=self.model,
            timesteps=config['diffusion']['timesteps'],
            beta_schedule=config['diffusion']['beta_schedule'],
            loss_type=config['diffusion']['loss_type']
        )
        
        # 初始化优化器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # 混合精度训练
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 初始化数据加载器
        self.train_loader = self._build_dataloader('train')
        self.val_loader = self._build_dataloader('val')
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 加载检查点
        if config['training'].get('resume_from'):
            self.load_checkpoint(config['training']['resume_from'])
    
    def _build_model(self):
        """构建模型"""
        model_config = self.config['model']
        model = LightweightUNet(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            channels=model_config['channels'],
            time_emb_dim=model_config['time_emb_dim'],
            num_res_blocks=model_config['num_res_blocks']
        ).to(self.device)
        
        # 多 GPU 训练
        if self.config['training'].get('distributed'):
            model = nn.DataParallel(model)
        
        # 打印模型信息
        params = model.count_parameters() if hasattr(model, 'count_parameters') else {}
        if params:
            print(f"Model parameters: {params['total']:,}")
            print(f"Model size: {params['total_mb']:.2f} MB")
        
        return model
    
    def _build_optimizer(self):
        """构建优化器"""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['type'] == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
        
        return optimizer
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        sched_config = self.config['scheduler']
        
        if sched_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config.get('eta_min', 0)
            )
        elif sched_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _build_dataloader(self, split):
        """构建数据加载器"""
        data_config = self.config['data']
        
        dataset = ImageDataset(
            root_dir=data_config[f'{split}_dir'],
            image_size=data_config['image_size']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=data_config['batch_size'],
            shuffle=(split == 'train'),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )
        
        return dataloader
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    loss = self.diffusion(images)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.diffusion(images)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 更新统计
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # TensorBoard 记录
            if self.global_step % self.config['training']['log_freq'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # 生成样本
            if self.global_step % self.config['training']['sample_freq'] == 0:
                self.sample_images()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        for images in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            loss = self.diffusion(images)
            total_loss += loss.item()
        
        val_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('val/loss', val_loss, self.epoch)
        
        return val_loss
    
    @torch.no_grad()
    def sample_images(self):
        """生成样本图像"""
        self.model.eval()
        
        # DDIM 采样
        samples = self.diffusion.ddim_sample(
            batch_size=4,
            image_size=self.config['data']['image_size'],
            ddim_timesteps=50,
            progress=False
        )
        
        # 保存图像
        samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        save_path = self.output_dir / 'samples' / f'step_{self.global_step}.png'
        save_path.parent.mkdir(exist_ok=True)
        
        # 转换为图像并保存
        from torchvision.utils import save_image
        save_image(samples, save_path, nrow=2)
        
        # TensorBoard 记录
        self.writer.add_images('samples', samples, self.global_step)
        
        self.model.train()
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
        
        # 定期保存
        if self.epoch % self.config['training']['save_freq'] == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{self.epoch}.pt')
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """主训练循环"""
        print(f"Starting training for {self.config['training']['epochs']} epochs")
        
        for epoch in range(self.epoch, self.config['training']['epochs']):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 打印统计
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(is_best)
        
        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train LightDiffusion model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖配置
    if args.device:
        config['device'] = args.device
    if args.resume:
        config['training']['resume_from'] = args.resume
    
    # 开始训练
    trainer = Trainer(config)
    trainer.train()


# 配置文件示例 - configs/default.yaml
DEFAULT_CONFIG = """
# 模型配置
model:
  in_channels: 3
  out_channels: 3
  channels: [64, 128, 256, 512]
  time_emb_dim: 256
  num_res_blocks: 2

# 扩散模型配置
diffusion:
  timesteps: 1000
  beta_schedule: cosine
  loss_type: l2

# 数据配置
data:
  train_dir: dataset/train
  val_dir: dataset/val
  image_size: 256
  batch_size: 16
  num_workers: 4

# 优化器配置
optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.01

# 学习率调度器
scheduler:
  type: cosine
  T_max: 100
  eta_min: 0.000001

# 训练配置
training:
  epochs: 100
  use_amp: true
  distributed: false
  log_freq: 100
  sample_freq: 1000
  save_freq: 10
  resume_from: null

# 其他配置
device: cuda
output_dir: outputs
seed: 42
"""

if __name__ == "__main__":
    # 如果配置文件不存在，创建默认配置
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(DEFAULT_CONFIG)
        print(f"Created default config at {config_path}")
    
    # 开始训练
    main()