#!/usr/bin/env python
"""
LightDiffusion 一键安装配置脚本
自动创建项目结构并安装依赖
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, shell=False):
    """运行系统命令"""
    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def create_directory_structure():
    """创建项目目录结构"""
    print("Creating directory structure...")
    
    directories = [
        'models',
        'data', 
        'training',
        'inference',
        'app',
        'configs',
        'utils',
        'weights',
        'outputs',
        'examples',
        'dataset/train',
        'dataset/val',
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # 创建 __init__.py 文件
        if not dir_path.startswith(('dataset', 'weights', 'outputs', 'examples')):
            init_file = Path(dir_path) / '__init__.py'
            init_file.touch()
    
    print("✓ Directory structure created")

def install_dependencies():
    """安装项目依赖"""
    print("\nInstalling dependencies...")
    
    # 基础依赖
    base_deps = [
        'numpy',
        'pillow',
        'opencv-python',
        'tqdm',
        'pyyaml',
        'matplotlib',
        'scikit-image',
        'scipy',
        'gradio',
        'fastapi',
        'uvicorn',
        'tensorboard',
        'albumentations',
        'kornia'
    ]
    
    # 检查 PyTorch 是否已安装
    try:
        import torch
        print("✓ PyTorch already installed")
    except ImportError:
        print("Installing PyTorch...")
        # 根据系统选择合适的 PyTorch 版本
        cuda_available = shutil.which('nvidia-smi') is not None
        
        if cuda_available:
            # CUDA 版本
            pytorch_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'torch', 'torchvision',
                '-f', 'https://download.pytorch.org/whl/torch_stable.html'
            ]
        else:
            # CPU 版本
            pytorch_cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision']
        
        success, output = run_command(pytorch_cmd)
        if success:
            print("✓ PyTorch installed")
        else:
            print(f"✗ Failed to install PyTorch: {output}")
            return False
    
    # 安装其他依赖
    for dep in base_deps:
        print(f"Installing {dep}...")
        success, output = run_command([sys.executable, '-m', 'pip', 'install', dep])
        if not success:
            print(f"Warning: Failed to install {dep}")
    
    # 可选依赖
    optional_deps = ['lpips', 'pytorch-fid', 'cleanfid']
    
    print("\nInstalling optional dependencies...")
    for dep in optional_deps:
        success, _ = run_command([sys.executable, '-m', 'pip', 'install', dep])
        if success:
            print(f"✓ {dep} installed")
        else:
            print(f"⚠ {dep} not installed (optional)")
    
    return True

def download_sample_data():
    """下载示例数据"""
    print("\nSetting up sample data...")
    
    # 创建示例图像（使用 numpy 生成）
    try:
        import numpy as np
        from PIL import Image
        
        # 创建一些示例图像
        for i in range(5):
            # 创建随机彩色图像
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # 保存到训练集
            img.save(f'dataset/train/sample_{i:03d}.jpg')
            
            # 创建带噪声的版本作为示例
            noise = np.random.normal(0, 25, (256, 256, 3))
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            Image.fromarray(noisy_img).save(f'examples/noisy_{i:03d}.jpg')
        
        # 创建验证集图像
        for i in range(2):
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(f'dataset/val/sample_{i:03d}.jpg')
        
        print("✓ Sample images created")
        
    except Exception as e:
        print(f"⚠ Could not create sample images: {e}")
    
    return True

def create_config_file():
    """创建默认配置文件"""
    print("\nCreating configuration file...")
    
    config_content = """# LightDiffusion Configuration

model:
  in_channels: 3
  out_channels: 3
  channels: [64, 128, 256, 512]
  time_emb_dim: 256
  num_res_blocks: 2

diffusion:
  timesteps: 1000
  beta_schedule: cosine
  loss_type: l2

data:
  train_dir: dataset/train
  val_dir: dataset/val
  image_size: 256
  batch_size: 8
  num_workers: 4

optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.01

scheduler:
  type: cosine_warmup
  warmup_steps: 1000
  max_steps: 10000
  min_lr: 0.000001

training:
  epochs: 100
  use_amp: true
  distributed: false
  log_freq: 100
  sample_freq: 1000
  save_freq: 10

device: cuda
output_dir: outputs
seed: 42
"""
    
    with open('configs/default.yaml', 'w') as f:
        f.write(config_content)
    
    print("✓ Configuration file created")

def create_main_script():
    """创建主入口脚本"""
    print("\nCreating main script...")
    
    main_content = '''#!/usr/bin/env python
"""
LightDiffusion 主入口脚本
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='LightDiffusion')
    parser.add_argument('--mode', type=str, choices=['train', 'demo', 'test', 'api'],
                       default='demo', help='运行模式')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    if args.mode == 'train':
        from training.train import main as train_main
        train_main()
    elif args.mode == 'demo':
        from app.gradio_app import create_interface
        demo = create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860)
    elif args.mode == 'api':
        import uvicorn
        from app.api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.mode == 'test':
        import test_all
        test_all.run_all_tests()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open('main.py', 'w') as f:
        f.write(main_content)
    
    print("✓ Main script created")

def create_readme():
    """创建 README 文件"""
    print("\nCreating README...")
    
    readme_content = """# LightDiffusion - 轻量级扩散模型图像修复系统

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 快速开始

### 1. 安装
```bash
python setup.py
```

### 2. 测试安装
```bash
python test_all.py
```

### 3. 运行演示
```bash
python main.py --mode demo
```

访问 http://localhost:7860

### 4. 训练模型
```bash
python main.py --mode train
```

## 功能特性

- 🚀 轻量级架构，参数量减少40%
- 🎨 支持图像去噪、超分辨率、对象移除
- 💻 友好的 Web 界面
- 🔧 模块化设计，易于扩展

## 项目结构

```
LightDiffusion/
├── models/         # 模型架构
├── data/          # 数据处理
├── training/      # 训练相关
├── inference/     # 推理模块
├── app/          # Web 应用
├── configs/      # 配置文件
└── utils/        # 工具函数
```

## 作者

李新宝 - lixinbao@njust.edu.cn

## 许可证

MIT License
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✓ README created")

def setup_git():
    """设置 Git 仓库"""
    print("\nSetting up Git repository...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Project specific
outputs/
weights/*.pt
weights/*.pth
dataset/
*.log
*.png
*.jpg
*.jpeg
!examples/*.jpg
!examples/*.png

# ML
*.ckpt
*.pth
tensorboard/
wandb/
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    # 初始化 Git 仓库
    if not Path('.git').exists():
        success, _ = run_command(['git', 'init'])
        if success:
            print("✓ Git repository initialized")
        else:
            print("⚠ Could not initialize Git repository")
    
    return True

def print_next_steps():
    """打印后续步骤"""
    print("\n" + "="*60)
    print(" 🎉 Setup Complete! ")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("\n1. Test the installation:")
    print("   python test_all.py")
    
    print("\n2. Run the web demo:")
    print("   python main.py --mode demo")
    print("   Open http://localhost:7860")
    
    print("\n3. Start training:")
    print("   python main.py --mode train")
    
    print("\n4. Run the API server:")
    print("   python main.py --mode api")
    print("   API docs at http://localhost:8000/docs")
    
    print("\n5. Add your own data:")
    print("   Copy images to dataset/train/ and dataset/val/")
    
    print("\n📚 Documentation:")
    print("   See README.md for detailed usage")
    
    print("\n💡 Tips:")
    print("   - Reduce batch_size in configs/default.yaml if you have limited GPU memory")
    print("   - Use --mode test to verify everything is working")
    print("   - Check outputs/ directory for training results")

def main():
    """主函数"""
    print("="*60)
    print(" LightDiffusion Setup Script ")
    print("="*60)
    
    # 确认安装
    response = input("\nThis will set up the LightDiffusion project. Continue? [Y/n]: ")
    if response.lower() == 'n':
        print("Setup cancelled.")
        return
    
    steps = [
        ("Creating directory structure", create_directory_structure),
        ("Installing dependencies", install_dependencies),
        ("Creating configuration", create_config_file),
        ("Creating main script", create_main_script),
        ("Creating README", create_readme),
        ("Setting up sample data", download_sample_data),
        ("Setting up Git", setup_git),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            step_func()
        except Exception as e:
            print(f"✗ Error in {step_name}: {e}")
            response = input("Continue anyway? [Y/n]: ")
            if response.lower() == 'n':
                print("Setup interrupted.")
                return
    
    print_next_steps()
    
    # 询问是否运行测试
    response = input("\n\nRun tests now? [Y/n]: ")
    if response.lower() != 'n':
        print("\nRunning tests...")
        subprocess.run([sys.executable, 'test_all.py'])

if __name__ == "__main__":
    main()
