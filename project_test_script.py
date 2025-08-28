#!/usr/bin/env python
"""
LightDiffusion 项目完整性测试脚本
运行此脚本以验证所有模块是否正确安装和配置
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def test_imports():
    """测试所有必要的库导入"""
    print("\n" + "="*50)
    print("测试库导入...")
    print("="*50)
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'cv2',
        'yaml',
        'tqdm',
        'matplotlib',
        'gradio',
        'fastapi',
        'sklearn',
    ]
    
    optional_packages = [
        'lpips',
        'tensorboard',
        'wandb',
    ]
    
    failed = []
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} 已安装")
        except ImportError:
            print_error(f"{package} 未安装")
            failed.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            print_success(f"{package} 已安装（可选）")
        except ImportError:
            print_warning(f"{package} 未安装（可选）")
    
    if failed:
        print_error(f"\n请安装缺失的包: pip install {' '.join(failed)}")
        return False
    return True

def test_project_structure():
    """测试项目目录结构"""
    print("\n" + "="*50)
    print("测试项目结构...")
    print("="*50)
    
    required_dirs = [
        'models',
        'data',
        'training',
        'inference',
        'app',
        'configs',
        'utils',
        'dataset',
        'dataset/train',
        'dataset/val',
    ]
    
    required_files = [
        'models/unet.py',
        'models/diffusion.py',
        'models/vae.py',
        'models/scheduler.py',
        'data/dataset.py',
        'data/preprocessing.py',
        'training/train.py',
        'training/losses.py',
        'training/optimizer.py',
        'utils/metrics.py',
        'configs/default.yaml',
        'requirements.txt',
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print_success(f"目录存在: {dir_path}")
        else:
            print_warning(f"目录缺失: {dir_path}")
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"文件存在: {file_path}")
        else:
            print_warning(f"文件缺失: {file_path}")
            missing_files.append(file_path)
    
    # 创建缺失的目录
    if missing_dirs:
        print_info("\n创建缺失的目录...")
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print_success(f"已创建: {dir_path}")
    
    return len(missing_files) == 0

def test_model_loading():
    """测试模型加载"""
    print("\n" + "="*50)
    print("测试模型架构...")
    print("="*50)
    
    try:
        import torch
        from models.unet import LightweightUNet
        from models.diffusion import GaussianDiffusion
        from models.vae import VAE
        from models.scheduler import DDPMScheduler, DDIMScheduler
        
        # 测试 UNet
        model = LightweightUNet(
            in_channels=3,
            out_channels=3,
            channels=[32, 64, 128, 256],  # 使用较小的模型进行测试
        )
        print_success("UNet 模型创建成功")
        
        # 测试参数量
        params = model.count_parameters()
        print_info(f"模型参数量: {params['total']:,} ({params['total_mb']:.2f} MB)")
        
        # 测试前向传播
        x = torch.randn(1, 3, 128, 128)
        t = torch.tensor([100]).float()
        with torch.no_grad():
            out = model(x, t)
        assert out.shape == x.shape
        print_success("UNet 前向传播测试通过")
        
        # 测试扩散模型
        diffusion = GaussianDiffusion(
            model=model,
            timesteps=100,  # 使用较少的步数进行测试
            beta_schedule="linear"
        )
        print_success("扩散模型创建成功")
        
        # 测试 VAE
        vae = VAE(
            in_channels=3,
            latent_channels=4,
            channels_mult=(1, 2, 2, 2)
        )
        print_success("VAE 模型创建成功")
        
        # 测试调度器
        ddpm = DDPMScheduler(num_train_timesteps=100)
        ddim = DDIMScheduler(num_train_timesteps=100)
        print_success("调度器创建成功")
        
        return True
        
    except Exception as e:
        print_error(f"模型测试失败: {str(e)}")
        return False

def test_data_pipeline():
    """测试数据管道"""
    print("\n" + "="*50)
    print("测试数据管道...")
    print("="*50)
    
    try:
        import torch
        import numpy as np
        from data.dataset import BaseImageDataset, InpaintingDataset
        from data.preprocessing import ImagePreprocessor, NoiseGenerator, MaskGenerator
        
        # 测试预处理器
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        processed = preprocessor.preprocess_image(test_image)
        assert processed.shape == (1, 3, 128, 128)
        print_success("图像预处理器测试通过")
        
        # 测试噪声生成
        noise = NoiseGenerator.gaussian_noise((1, 3, 128, 128))
        assert noise.shape == (1, 3, 128, 128)
        print_success("噪声生成器测试通过")
        
        # 测试掩码生成
        mask = MaskGenerator.irregular_mask(128, 128)
        assert mask.shape == (1, 1, 128, 128)
        print_success("掩码生成器测试通过")
        
        # 测试数据集（如果有数据）
        if os.path.exists('dataset/train') and len(os.listdir('dataset/train')) > 0:
            dataset = BaseImageDataset('dataset/train', image_size=128)
            print_success(f"数据集加载成功，包含 {len(dataset)} 张图像")
            
            if len(dataset) > 0:
                sample = dataset[0]
                assert sample.shape == (3, 128, 128)
                print_success("数据集采样测试通过")
        else:
            print_warning("dataset/train 目录为空，跳过数据集测试")
        
        return True
        
    except Exception as e:
        print_error(f"数据管道测试失败: {str(e)}")
        return False

def test_training_components():
    """测试训练组件"""
    print("\n" + "="*50)
    print("测试训练组件...")
    print("="*50)
    
    try:
        import torch
        import torch.nn as nn
        from training.losses import DiffusionLoss, PerceptualLoss, TotalVariationLoss
        from training.optimizer import create_optimizer, create_scheduler
        
        # 创建简单模型
        model = nn.Linear(10, 10)
        
        # 测试优化器
        optimizer = create_optimizer(model, 'adam', lr=0.001)
        print_success("优化器创建成功")
        
        # 测试调度器
        scheduler = create_scheduler(optimizer, 'cosine', T_max=100)
        print_success("学习率调度器创建成功")
        
        # 测试损失函数
        diff_loss = DiffusionLoss(loss_type='l2')
        tv_loss = TotalVariationLoss()
        
        test_pred = torch.randn(2, 3, 64, 64)
        test_target = torch.randn(2, 3, 64, 64)
        
        loss1 = diff_loss(test_pred, test_target)
        loss2 = tv_loss(test_pred)
        
        assert loss1.item() >= 0
        assert loss2.item() >= 0
        print_success("损失函数测试通过")
        
        return True
        
    except Exception as e:
        print_error(f"训练组件测试失败: {str(e)}")
        return False

def test_inference():
    """测试推理功能"""
    print("\n" + "="*50)
    print("测试推理功能...")
    print("="*50)
    
    try:
        import torch
        import numpy as np
        from models.unet import LightweightUNet
        from models.diffusion import GaussianDiffusion, InpaintingDiffusion
        
        # 创建小模型进行测试
        model = LightweightUNet(channels=[16, 32, 64, 128])
        model.eval()
        
        # 测试 DDIM 采样
        diffusion = GaussianDiffusion(model, timesteps=100)
        
        with torch.no_grad():
            # 快速测试（步数很少）
            samples = diffusion.ddim_sample(
                batch_size=1,
                image_size=64,
                ddim_timesteps=5,
                progress=False
            )
            assert samples.shape == (1, 3, 64, 64)
            print_success("DDIM 采样测试通过")
        
        # 测试图像修复
        inpaint_diff = InpaintingDiffusion(model, timesteps=100)
        
        image = torch.randn(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1
        
        with torch.no_grad():
            result = inpaint_diff.inpaint(
                image, mask,
                ddim_steps=5,
                progress=False
            )
            assert result.shape == image.shape
            print_success("图像修复测试通过")
        
        return True
        
    except Exception as e:
        print_error(f"推理测试失败: {str(e)}")
        return False

def test_metrics():
    """测试评估指标"""
    print("\n" + "="*50)
    print("测试评估指标...")
    print("="*50)
    
    try:
        import numpy as np
        from utils.metrics import MetricsCalculator, MetricTracker
        
        calculator = MetricsCalculator()
        
        # 创建测试图像
        pred = np.random.rand(64, 64, 3).astype(np.float32)
        target = np.random.rand(64, 64, 3).astype(np.float32)
        
        # 计算指标
        psnr = calculator.calculate_psnr(pred, target)
        ssim = calculator.calculate_ssim(pred, target)
        mae = calculator.calculate_mae(pred, target)
        
        assert psnr > 0
        assert 0 <= ssim <= 1
        assert mae >= 0
        
        print_success(f"PSNR: {psnr:.2f} dB")
        print_success(f"SSIM: {ssim:.4f}")
        print_success(f"MAE: {mae:.4f}")
        
        # 测试跟踪器
        tracker = MetricTracker(['psnr', 'ssim'])
        tracker.update({'psnr': psnr, 'ssim': ssim})
        
        current = tracker.get_current()
        assert 'psnr' in current and 'ssim' in current
        print_success("指标跟踪器测试通过")
        
        return True
        
    except Exception as e:
        print_error(f"指标测试失败: {str(e)}")
        return False

def test_gpu():
    """测试 GPU 可用性"""
    print("\n" + "="*50)
    print("测试 GPU 配置...")
    print("="*50)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print_success(f"CUDA 可用，检测到 {gpu_count} 个 GPU")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print_info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 测试 GPU 操作
            x = torch.randn(1, 3, 128, 128).cuda()
            y = x * 2
            assert y.is_cuda
            print_success("GPU 计算测试通过")
            
        else:
            print_warning("CUDA 不可用，将使用 CPU 进行训练（速度会很慢）")
            
    except Exception as e:
        print_warning(f"GPU 测试失败: {str(e)}")
    
    return True

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print(" LightDiffusion 项目完整性测试 ")
    print("="*60)
    
    tests = [
        ("依赖库", test_imports),
        ("项目结构", test_project_structure),
        ("模型架构", test_model_loading),
        ("数据管道", test_data_pipeline),
        ("训练组件", test_training_components),
        ("推理功能", test_inference),
        ("评估指标", test_metrics),
        ("GPU 配置", test_gpu),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"{test_name} 测试出现异常: {str(e)}")
            results[test_name] = False
    
    # 打印总结
    print("\n" + "="*60)
    print(" 测试总结 ")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}通过{Colors.END}" if passed else f"{Colors.RED}失败{Colors.END}"
        print(f"{test_name}: {status}")
    
    print("-"*60)
    
    if passed == total:
        print_success(f"所有测试通过！({passed}/{total})")
        print_info("\n项目已准备就绪！你可以开始：")
        print("  1. 运行 Web 界面: python app/gradio_app.py")
        print("  2. 开始训练: python training/train.py")
        print("  3. 运行 API: python app/api.py")
    else:
        print_warning(f"部分测试未通过 ({passed}/{total})")
        print_info("\n请检查失败的测试并修复问题")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)