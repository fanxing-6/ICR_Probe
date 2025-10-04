"""
ICR探针的配置模块。

该模块定义了训练ICR探针所需的所有配置参数，包括：
- 模型架构参数
- 训练超参数
- 学习率调度参数
- 数据处理参数
- 文件路径配置
"""

from dataclasses import dataclass
import argparse
from typing import Optional

@dataclass
class Config:
    """
    ICR探针训练的配置类。
    
    使用dataclass装饰器自动生成初始化方法和其他常用方法。
    所有配置参数都提供了默认值，可以通过命令行参数覆盖。
    """
    
    # 模型结构参数
    input_dim: int = 32      # 输入特征维度
    hidden_dim: int = 128    # 隐藏层维度
    
    # 训练相关参数
    batch_size: int = 16     # 批处理大小
    num_epochs: int = 100    # 训练轮数
    learning_rate: float = 1e-3   # 初始学习率
    weight_decay: float = 1e-5    # 权重衰减系数（L2正则化）
    
    # 学习率调度参数
    lr_factor: float = 0.5   # 学习率衰减因子
    lr_patience: int = 5     # 学习率衰减等待轮数（验证损失未改善）
    
    # 数据集相关参数
    test_size: float = 0.2   # 测试集比例
    dataset_weight: bool = True  # 是否使用数据集权重
    
    # 文件路径配置
    data_dir: str = None    # 数据集目录路径
    save_dir: str = None    # 模型保存目录路径
    
    @classmethod
    def from_args(cls):
        """
        从命令行参数创建配置对象。
        
        使用argparse解析命令行参数，目前支持的参数：
        --data_dir: 数据集目录路径（必需）
        --save_dir: 模型保存目录路径（必需）
        
        Returns:
            Config: 根据命令行参数创建的配置对象
        """
        parser = argparse.ArgumentParser(description='ICR探针训练配置参数')
        # 添加命令行参数
        parser.add_argument('--data_dir', required=True, help='数据集目录的路径')
        parser.add_argument('--save_dir', required=True, help='模型保存目录的路径')
        
        args = parser.parse_args()
        return cls(**vars(args))
