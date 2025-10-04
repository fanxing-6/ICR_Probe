# -*- coding: utf-8 -*-
"""
ICR探针训练器模块

这个模块实现了用于训练ICR（Interpretable Chain-of-thought Reasoning）
探针的训练器类。该探针用于分析和评估大型语言模型的推理链质量。
"""

import os
import json
import logging
from typing import List, Dict, Union, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, auc, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import ICRProbe


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICRProbeTrainer:
    """
    ICR探针训练器类
    
    该类负责ICR探针模型的训练、验证和评估过程。包括：
    - 数据加载和预处理
    - 模型训练和验证
    - 性能指标计算和监控
    - 模型保存和加载
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        初始化训练器。

        Args:
            model: ICR探针模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置参数对象，包含学习率等超参数
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def setup_data(self):
        """Setup data loaders."""
        data = self._load_data()
        self.train_loader, self.val_loader = self._create_data_loaders(data)
        
    def setup_model(self):
        """Setup model and optimization components."""
        input_dim = next(iter(self.train_loader))[0].shape[1]
        self.model = ICRProbe(input_dim=input_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.lr_factor,
            patience=self.config.lr_patience
        )

    def train(self):
        """Train the model."""
        best_val_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch()
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_metrics)
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_model()
                
            self.scheduler.step(val_metrics['val_loss'])

    def save_model(self):
        """Save model and configuration."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), save_dir / 'model.pth')
        
        # Save config
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

    def _train_epoch(self):
        """
        执行一个训练周期。
        
        包括以下步骤：
        1. 将模型设置为训练模式
        2. 遍历训练数据批次
        3. 前向传播计算损失
        4. 反向传播更新参数
        
        Returns:
            float: 该周期的平均训练损失
        """
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in self.train_loader:
            # 将数据移动到指定设备
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # 清零梯度，前向传播，计算损失
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch.unsqueeze(1))
            # 反向传播，更新参数
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        """
        执行一个验证周期并计算各项评估指标。
        
        计算的指标包括：
        - 验证损失
        - 准确率、精确率、召回率
        - F1分数
        - ROC-AUC曲线
        - 皮尔逊相关系数（PCC）
        - 最优阈值
        - 正负样本的平均预测值
        
        Returns:
            Dict: 包含所有计算得到的评估指标
        """
        self.model.eval()
        metrics = {}
        val_losses = []
        val_preds = []
        val_preds_continuous = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.unsqueeze(1))
                val_losses.append(loss.item())
                outputs = outputs.squeeze()
                preds = outputs.round()
                preds_continuous = outputs
                val_preds.extend(preds.cpu().numpy().tolist())
                val_preds_continuous.extend(preds_continuous.cpu().numpy().tolist())
                val_labels.extend(y_batch.cpu().numpy().tolist())

        TP = FP = FN = TN = 0
        for i in range(len(val_preds_continuous)):
            if val_preds_continuous[i] >= self.config.halu_threshold:
                if val_labels[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if val_labels[i] == 1:
                    FN += 1
                else:
                    TN += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        mean_neg_pred = np.mean([val_preds_continuous[i] for i in range(len(val_preds_continuous)) if val_labels[i] == 0])
        mean_pos_pred = np.mean([val_preds_continuous[i] for i in range(len(val_preds_continuous)) if val_labels[i] == 1])
        f1 = f1_score(val_labels, val_preds)
        fpr, tpr, thresholds = roc_curve(val_labels, val_preds_continuous)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        pcc = np.corrcoef(val_labels, val_preds_continuous)[0, 1]
        avg_val_loss = np.mean(val_losses)

        metrics['val_loss'] = avg_val_loss
        metrics['Precision'] = precision
        metrics['Recall'] = recall
        metrics['Accuracy'] = accuracy
        metrics['F1 Score'] = f1
        metrics['ROC-AUC'] = roc_auc
        metrics['PCC'] = pcc
        metrics['optimal_threshold'] = optimal_threshold
        metrics['mean_neg_pred'] = mean_neg_pred
        metrics['mean_pos_pred'] = mean_pos_pred

        return metrics

    def _log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict):
        """Log training and validation metrics."""
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        for name, value in val_metrics.items():
            logger.info(f"{name}: {value:.4f}")

