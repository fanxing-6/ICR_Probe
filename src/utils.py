"""
工具模块，包含ICR探针的神经网络模型定义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ICRProbe(nn.Module):
    """
    ICR探针的多层感知机分类器。
    
    网络结构：
    输入层 [input_dim] -> 128 -> 64 -> 32 -> 1 (输出层)
    
    特点：
    - 使用BatchNorm进行归一化
    - 使用Dropout防止过拟合
    - 使用LeakyReLU作为激活函数
    - 最后使用Sigmoid函数输出0-1之间的概率值
    """
    def __init__(self, input_dim=32):
        """
        初始化ICR探针网络。
        
        Args:
            input_dim: 输入特征的维度，默认为32
            
        网络各层说明：
        - fc1: 第一个全连接层，将输入维度映射到128
        - bn1: 第一个批归一化层，用于归一化特征
        - dropout1: 第一个dropout层，丢弃率为0.3
        """
        super(ICRProbe, self).__init__()

        # 第一层: 输入维度 -> 128
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        # 第二层: 128 -> 64
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """
        初始化网络的权重和偏置。
        
        初始化策略：
        - 线性层：使用Kaiming初始化（He初始化），适合LeakyReLU激活函数
        - 批归一化层：权重初始化为1，偏置初始化为0
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        前向传播函数。
        
        Args:
            x: 输入张量，形状为[batch_size, input_dim]
            
        Returns:
            输出张量，形状为[batch_size, 1]，值域为[0,1]
            
        Note:
            每一层的处理顺序：线性层 -> 批归一化 -> LeakyReLU -> Dropout
        """
        # 检查输入是否含有NaN值
        assert torch.isnan(x).sum() == 0, f"Input contains NaN values:{torch.isnan(x).sum()}"
        
        # 第一层处理
        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        out = self.dropout1(out)

        out = F.leaky_relu(self.bn2(self.fc2(out)), negative_slope=0.01)
        out = self.dropout2(out)

        out = F.leaky_relu(self.bn3(self.fc3(out)), negative_slope=0.01)
        out = self.dropout3(out)

        out = self.sigmoid(self.fc4(out))
        return out
    
    