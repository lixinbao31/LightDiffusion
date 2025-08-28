# 配置文件使用指南

## 📁 配置文件结构

```
configs/
├── default.yaml           # 默认配置（完整参数）
├── small_model.yaml      # 小模型（快速实验）
├── large_model.yaml      # 大模型（最佳质量）
├── low_gpu_memory.yaml   # 低显存优化
├── fast_training.yaml    # 快速训练
├── production.yaml       # 生产部署
├── research.yaml         # 研究实验
├── denoise_specific.yaml # 去噪专用
├── super_resolution.yaml # 超分专用
└── inpainting.yaml       # 修复专用
```

## 🚀 快速选择指南

### 根据硬件选择

| GPU 显存 | 推荐配置 | 批次大小 | 图像尺寸 |
|---------|---------|---------|---------|
| 4GB | `low_gpu_memory.yaml` | 2 | 128 |
| 8GB | `small_model.yaml` | 8 | 128 |
| 12GB | `default.yaml` | 16 | 256 |
| 24GB+ | `large_model.yaml` | 32 | 512 |

### 根据目的选择

- **初次尝试**: `small_model.yaml` - 快速看到结果
- **正式训练**: `default.yaml` - 平衡的配置
- **发表论文**: `research.yaml` - 完整的实验设置
- **部署服务**: `production.yaml` - 优化的推理配置
- **快速原型**: `fast_training.yaml` - 最快获得模型

## 🎯 使用方法

### 1. 基础使用

```bash
# 使用默认配置
python training/train.py --config configs/default.yaml

# 使用特定配置
python training/train.py --config configs/small_model.yaml

# 覆盖配置参数
python training/train.py --config configs/default.yaml \
    --batch_size 8 \
    --lr 0.0001
```

### 2. 创建自定义配置

创建 `configs/custom.yaml`:

```yaml
# 基于默认配置，只修改需要的部分
model:
  channels: [48, 96, 192, 384]  # 自定义通道数

data:
  batch_size: 12                # 自定义批次
  image_size: 192               # 自定义尺寸

training:
  epochs: 75                    # 自定义轮数
```

### 3. 配置继承

可以基于现有配置创建新配置：

```python
# load_config.py
import yaml

def load_config_with_override(base_config, custom_config):
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    if custom_config:
        with open(custom_config, 'r') as f:
            custom = yaml.safe_load(f)
        # 递归更新配置
        deep_update(config, custom)
    
    return config
```

## 🔧 参数调优建议

### 显存不足时的调整策略

1. **减小批次大小**
   ```yaml
   data:
     batch_size: 4  # 从16减到4
   ```

2. **使用梯度累积**
   ```yaml
   training:
     gradient_accumulation_steps: 4  # 等效于 batch_size * 4
   ```

3. **启用梯度检查点**
   ```yaml
   model:
     use_checkpoint: true  # 用时间换空间
   ```

4. **减小模型规模**
   ```yaml
   model:
     channels: [32, 64, 128, 256]  # 减半通道数
   ```

5. **使用混合精度**
   ```yaml
   training:
     use_amp: true  # FP16 训练
   ```

### 加速训练的策略

1. **增大学习率和批次**
   ```yaml
   optimizer:
     lr: 0.001  # 更大的学习率
   data:
     batch_size: 64  # 更大的批次
   ```

2. **减少验证频率**
   ```yaml
   training:
     validate_every: 5000  # 减少验证开销
     sample_every: 5000
   ```

3. **简化模型**
   ```yaml
   model:
     num_res_blocks: 1  # 减少残差块
   diffusion:
     timesteps: 500  # 减少扩散步数
   ```

### 提升生成质量

1. **增加模型容量**
   ```yaml
   model:
     channels: [128, 256, 512, 512]
     num_res_blocks: 3
   ```

2. **使用更多采样步数**
   ```yaml
   diffusion:
     sampling:
       ddim_steps: 100  # 更多步数
   ```

3. **添加辅助损失**
   ```yaml
   losses:
     perceptual_loss:
       enabled: true
       weight: 0.1
   ```

## 📊 配置参数详解

### 关键参数影响

| 参数 | 影响 | 建议范围 |
|-----|------|---------|
| `channels` | 模型容量 | [32-128, 64-256, 128-512, 256-1024] |
| `timesteps` | 生成质量 vs 速度 | 500-1000 |
| `batch_size` | 训练速度 vs 显存 | 2-64 |
| `lr` | 收敛速度 vs 稳定性 | 1e-5 - 1e-3 |
| `image_size` | 分辨率 vs 显存 | 64-512 |
| `ddim_steps` | 推理质量 vs 速度 | 20-100 |

### 任务特定优化

**去噪任务**：
- 使用 L1 损失
- 较少的扩散步数（500）
- 添加边缘损失

**超分辨率**：
- 使用感知损失
- 更多注意力层
- 保持细节的损失函数

**图像修复**：
- 输入通道数 +1（掩码）
- 条件生成
- 风格一致性损失

## 🔍 配置验证

运行配置验证脚本：

```python
# validate_config.py
import yaml
import os

def validate_config(config_path):
    """验证配置文件的合法性"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 检查必要字段
    required = ['model', 'data', 'training', 'optimizer']
    for field in required:
        assert field in config, f"Missing required field: {field}"
    
    # 检查路径存在
    assert os.path.exists(config['data']['train_dir'])
    
    # 检查数值合理性
    assert config['data']['batch_size'] > 0
    assert config['optimizer']['lr'] > 0
    
    print(f"✓ Config {config_path} is valid!")

# 使用
validate_config('configs/default.yaml')
```

## 💡 最佳实践

1. **始终从小开始**：先用 `small_model.yaml` 验证流程
2. **保存配置快照**：训练时自动复制配置到输出目录
3. **版本控制**：将配置文件纳入 Git 管理
4. **记录实验**：为每个配置添加描述性注释
5. **参数搜索**：使用 `research.yaml` 进行超参数搜索

## 🛠️ 故障排除

### 常见问题

**问题**: CUDA out of memory
```yaml
# 解决方案：使用 low_gpu_memory.yaml 或调整：
data:
  batch_size: 2
model:
  use_checkpoint: true
training:
  gradient_accumulation_steps: 8
```

**问题**: 训练不稳定
```yaml
# 解决方案：降低学习率
optimizer:
  lr: 0.00001
scheduler:
  warmup_steps: 5000
```

**问题**: 生成质量差
```yaml
# 解决方案：增加模型容量和训练时间
model:
  channels: [128, 256, 512, 512]
training:
  epochs: 200
diffusion:
  sampling:
    ddim_steps: 100
```

## 📝 配置模板生成器

快速生成配置的 Python 脚本：

```python
# generate_config.py
def generate_config(task, gpu_memory, quality='balanced'):
    """根据需求生成配置"""
    
    configs = {
        'denoise': 'configs/denoise_specific.yaml',
        'super_resolution': 'configs/super_resolution.yaml',
        'inpainting': 'configs/inpainting.yaml'
    }
    
    if gpu_memory < 8:
        base = 'configs/low_gpu_memory.yaml'
    elif gpu_memory < 16:
        base = 'configs/small_model.yaml'
    else:
        base = 'configs/default.yaml'
    
    print(f"Recommended config: {base}")
    print(f"Task-specific config: {configs.get(task, 'default.yaml')}")
    
    return base

# 使用示例
config = generate_config(task='denoise', gpu_memory=8)
```

---

现在你有了完整的配置系统！选择合适的配置文件开始你的训练吧。