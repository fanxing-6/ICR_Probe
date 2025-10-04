"""
这个模块实现了ICR (Interpretable Chain-of-thought Reasoning) 评分机制，
用于分析和评估语言模型的推理过程。
"""

import torch
import torch.nn.functional as F
import time
import numpy as np

def move_tensors_to_device(container, device):
    """
    将张量或包含张量的容器（列表、元组、字典）移动到指定的设备上。
    
    Args:
        container: 要移动的张量或包含张量的容器
        device: 目标设备（CPU或GPU）
    
    Returns:
        移动到指定设备后的张量或容器
    """
    if isinstance(container, torch.Tensor):
        return container.to(device)
    elif isinstance(container, list):
        return [move_tensors_to_device(x, device) if isinstance(x, torch.Tensor) else x for x in container]
    elif isinstance(container, tuple):
        return tuple(move_tensors_to_device(x, device) if isinstance(x, torch.Tensor) else x for x in container)
    elif isinstance(container, dict):
        return {k: move_tensors_to_device(v, device) if isinstance(v, torch.Tensor) else v for k, v in container.items()}
    else:
        return container

class ICRScore:
    """
    ICR评分器类，用于计算和分析模型的解释性链式推理过程。
    
    该类实现了基于注意力机制和隐藏状态的分析方法，可以识别和评估模型的推理链。
    """

    def __init__(self, hidden_states, attentions, skew_threshold=3, entropy_threshold=3, core_positions=None,icr_device=None):
        """
        初始化ICR评分器。

        Args:
            hidden_states: 模型各层的隐藏状态
            attentions: 模型各层的注意力权重
            skew_threshold: 判断归纳头的偏度阈值
            entropy_threshold: 判断归纳头的熵阈值
            core_positions: 核心位置的字典，包含用户输入和响应的起始结束位置
            icr_device: 计算设备（CPU或GPU）
        """
        self.origional_device = hidden_states[0][0].device  # 原始设备
        self.icr_device = icr_device  # ICR计算使用的设备
        if self.icr_device != self.origional_device:
            hidden_states = move_tensors_to_device(hidden_states, self.icr_device)
            attentions = move_tensors_to_device(attentions, self.icr_device)

        self.input_lens = hidden_states[0][0].shape[1]
        self.core_positions = core_positions
        with torch.no_grad():
            self.origin_hidden_states = self._pre_process_hs(hidden_states)
            self.origin_attentions = self._pre_process_attn(attentions)  

        # [layer, tokens, hidden_size]
        self.output_hidden_states = self.origin_hidden_states[:, self.input_lens:]
        # [layer, head, input_size, input_size]
        self.output_attentions = self.origin_attentions[:, :, self.input_lens:]

        with torch.no_grad():
            self.induction_head = self._is_induction_head(skew_threshold=skew_threshold,
                                                      entropy_threshold=entropy_threshold)

    def _pre_process_hs(self, hidden_states):
        """
        预处理隐藏状态，将输入和输出的隐藏状态整合到一个统一的张量中。
        
        Args:
            hidden_states: 模型生成的原始隐藏状态列表
            
        Returns:
            处理后的隐藏状态张量，维度为[层数, 序列长度, 隐藏维度]
        """
        # 处理输入隐藏状态 [output_size, layer, batch_size,(input_size/1), hidden_size]
        hidden_states_input = torch.stack(hidden_states[0], dim=0)
        # 提取每层的输入状态 [layer, input_size, hidden_size]
        hs_input = hidden_states_input[:, 0, :]
        # 处理输出隐藏状态
        hidden_states_output = torch.stack([torch.stack(layer) for layer in hidden_states[1:]], dim=0)
        hs_output = torch.cat([hidden_states_output[i, :, 0] for i in range(len(hidden_states_output))], dim=1)
        hs_all = torch.cat([hs_input, hs_output], dim=1)  # shape: [layer, input_size + output_size, hidden_size]
        del hidden_states_input, hs_input, hidden_states_output, hs_output
        with torch.cuda.device(self.icr_device):
            torch.cuda.empty_cache()
        return hs_all

    def _pre_process_attn(self, attentions):
        """
        预处理注意力权重，将所有层和头的注意力分数整合成统一格式。
        
        Args:
            attentions: 原始注意力权重列表
            
        Returns:
            处理后的注意力权重张量，维度为[层数, 头数, 序列长度, 序列长度]
        """
        # 计算总token数，头数和层数
        token_num = len(attentions) + attentions[0][0][0].shape[1] - 1
        head_num = attentions[0][0][0].shape[0]
        layer_num = len(attentions[0])
        device = attentions[0][0][0].device
        dtype = attentions[0][0][0].dtype

        input_token_num = self.input_lens
        input_attn = []
        padding_size = token_num - input_token_num

        for layer_idx in range(layer_num):
            layer_attn = []
            for head_idx in range(head_num):

                padded_input_attn = F.pad(
                    attentions[0][layer_idx][0][head_idx, :input_token_num],
                    (0, padding_size)
                )
                layer_attn.append(padded_input_attn)
            input_attn.append(torch.stack(layer_attn, dim=0))
        input_attn = torch.stack(input_attn, dim=0)  # (layer_num, head_num, input_token_num, token_num)

        output_token_num = token_num - input_token_num
        output_attn = []

        for layer_idx in range(layer_num):
            layer_attn = []
            for head_idx in range(head_num):
     
                padded_output_attn = torch.stack([
                    F.pad(
                        attentions[token_idx - input_token_num + 1][layer_idx][0][head_idx, 0],
                        (0, token_num - token_idx - 1)
                    )
                    for token_idx in range(input_token_num, token_num)
                ])
                layer_attn.append(padded_output_attn)
            output_attn.append(torch.stack(layer_attn, dim=0))
        output_attn = torch.stack(output_attn, dim=0)  # (layer_num, head_num, output_token_num, token_num)
        attn_all_o = torch.cat([input_attn, output_attn], dim=2)  # (layer_num, head_num, token_num, token_num)

        # set_other_attn_scores_to_zero
        attn_all = self.set_other_attn_scores_to_zero(attn_all_o)

        del input_attn, output_attn, attn_all_o
        with torch.cuda.device(self.icr_device):
            torch.cuda.empty_cache()
        return attn_all

    def set_other_attn_scores_to_zero(self, attn_all):
        """
        Set the attention scores of positions other than the specified ones to 0.

        :param attn_all: A 4D tensor with shape (layer_num, head_num, token_num, token_num).
        :param a: Start position of user.
        :param b: End position of user.
        :param c: Start position.
        :return: The masked tensor with other attn scores set to 0.
        """
        layer_num, head_num, token_num, _ = attn_all.size()
        a, b, c = self.core_positions['user_prompt_start'], self.core_positions['user_prompt_end'], self.core_positions[
            'response_start']
        # Create a boolean mask where positions to keep are True and others are False
        mask = torch.zeros((layer_num, head_num, token_num, token_num), dtype=torch.bool)
        mask[:, :, a:b, a:b] = True
        mask[:, :, c:, c:] = True

        # Use the boolean mask to set unwanted positions to 0
        attn_all[~mask] = 0
        del mask
        with torch.cuda.device(self.icr_device):
            torch.cuda.empty_cache()
        return attn_all

    def _calculate_skewness_entropy(self, attn_map):
        """
        计算注意力图的偏度和熵值。
        
        Args:
            attn_map: 注意力权重图，维度为[序列长度, 序列长度]
            
        Returns:
            average_skewness: 平均偏度值
            average_entropy: 平均熵值
        """
        sequence_size = attn_map.size(0)
        # 计算行和并进行归一化
        row_sums = attn_map.sum(dim=1, keepdim=True)
        row_normalized = attn_map / (row_sums + 1e-12)  
        # 生成位置索引
        indices = torch.arange(1, sequence_size + 1, device=attn_map.device, dtype=attn_map.dtype).view(1, -1)

        mean_indices = (row_normalized * indices).sum(dim=1)

        variance = ((indices - mean_indices.unsqueeze(1)) ** 2 * row_normalized).sum(dim=1)
        third_moment = ((indices - mean_indices.unsqueeze(1)) ** 3 * row_normalized).sum(dim=1)
        skewness = third_moment / (variance ** 1.5 + 1e-12)

        entropy = -torch.sum(row_normalized * torch.log2(row_normalized + 1e-12), dim=1) # range: min=0, max=log2(sequence_size) 

        valid_rows = row_sums.squeeze() > 0  
        average_skewness = skewness[valid_rows].mean().item()
        average_entropy = entropy[valid_rows].mean().item()

        return average_skewness, average_entropy

    def _is_induction_head(self, skew_threshold, entropy_threshold):
        """
        识别模型中的归纳头（Induction Head）。
        
        归纳头是一种特殊的注意力头，它能够捕捉到序列中的重复模式。
        通过分析注意力头的偏度和熵来识别归纳头。
        
        Args:
            skew_threshold: 偏度阈值，高于此值的注意力头可能是归纳头
            entropy_threshold: 熵阈值，低于此值的注意力头可能是归纳头
            
        Returns:
            is_induction_layer_head: 每一层中识别出的归纳头列表
        """
        is_induction_layer_head = []
        skew_entropy_values = []
        idx = 0
        for layer_attentions in self.origin_attentions:  
            num_heads = layer_attentions.size(0)  
            # 初始化存储偏度和熵的张量
            skewness_entropy = torch.zeros(num_heads, 2, device=layer_attentions.device)

            for head_idx in range(num_heads):
                attn_map = layer_attentions[head_idx]
                skewness, entropy = self._calculate_skewness_entropy(attn_map)
                skewness_entropy[head_idx] = torch.tensor([skewness, entropy])

            skewness = skewness_entropy[:, 0]
            entropy = skewness_entropy[:, 1]

            # Induction Head
            is_induction_head = (skewness >= skew_threshold) & (entropy <= entropy_threshold)
            idx += 1
            # Induction Head Top-K Heads
            if is_induction_head.sum() < num_heads // 8:
                top_heads = skewness.topk(num_heads // 8, largest=True).indices
                is_induction_head[:] = False
                is_induction_head[top_heads] = True

            skew_entropy_values.append(skewness_entropy)
            is_induction_layer_head.append(is_induction_head.tolist())
        with torch.cuda.device(self.icr_device):
            torch.cuda.empty_cache()
        return is_induction_layer_head 

    def _pooling_attn(self, pooling,use_induction_head):
        pooled_attentions = []
        for layer_idx in range(len(self.output_attentions)):
            induction_heads_this_layer = []
            for head_idx in range(len(self.output_attentions[layer_idx])):
                if use_induction_head:
                    if self.induction_head[layer_idx][head_idx]:
                        induction_heads_this_layer.append(self.output_attentions[layer_idx][head_idx])
                else:
                    induction_heads_this_layer.append(self.output_attentions[layer_idx][head_idx])

            if induction_heads_this_layer:
                stacked_heads = torch.stack(induction_heads_this_layer)
                if pooling == 'mean':
                    pooled_layer = torch.mean(stacked_heads, dim=0)
                elif pooling == 'max':
                    pooled_layer = torch.max(stacked_heads, dim=0)[0]
                elif pooling == 'min':
                    pooled_layer = torch.min(stacked_heads, dim=0)[0]
                else:
                    raise ValueError(f"{pooling} is not a valid pooling method.")
                pooled_attentions.append(pooled_layer)
            else:
                input_size = self.output_attentions[layer_idx][0].shape[-2:] if self.output_attentions[layer_idx] else (
                    1, 1)
                pooled_attentions.append(torch.zeros(input_size))
                raise ValueError(f"Layer {layer_idx} has no induction head.")
        with torch.cuda.device(self.icr_device):
            torch.cuda.empty_cache()
        return pooled_attentions  # [layer, output_size, all_size]

    def compute_icr(self, top_k, top_p, pooling, attention_uniform, hidden_uniform, use_induction_head):
        """
        计算每一层每个token的ICR分数。
        
        Args:
            top_k: 选择前k个最高注意力值
            top_p: 选择累积概率达到p的注意力值
            pooling: 注意力头池化方式 ('mean', 'max', 'min')
            attention_uniform: 是否使用均匀注意力分布（用于消融实验）
            hidden_uniform: 是否使用均匀隐藏状态权重（用于消融实验）
            use_induction_head: 是否只使用归纳头
            
        Returns:
            icr_scores_item: 每层每个token的ICR分数
            top_p_mean: 平均top-p值
        """
        # 池化注意力分数 [layer, output_size, all_size]
        self.pooling_attentions = self._pooling_attn(pooling=pooling,use_induction_head=use_induction_head)
        icr_scores_item = []  
        top_p_list = []
        for layer in range(len(self.pooling_attentions)):
            icr_scores_layer = []
            top_p_layer = []
            for token in range(len(self.pooling_attentions[layer])):

                current_token_attn = self.pooling_attentions[layer][token]

 
                top_k = min(top_k, len(current_token_attn)) if (top_k is not None) else len(current_token_attn)
                top_k = top_k if top_p is None else int(top_p * len(current_token_attn))
                top_p_token = top_k/max(len(current_token_attn),1e-6)
                top_p_layer.append(top_p_token)
                current_token_attn_topk, current_token_attn_topk_idx = torch.topk(current_token_attn, k=top_k)
             
                current_token_hs = self.output_hidden_states[layer + 1][token]
                previous_token_hs = self.output_hidden_states[layer][token]
                current_layer_all_hs = self.origin_hidden_states[layer]
                current_token_hs_topk = current_layer_all_hs[current_token_attn_topk_idx]
                hs_diff = (current_token_hs - previous_token_hs)
    
                w_i = torch.sum(hs_diff * current_token_hs_topk, dim=1) / (
                        torch.norm(current_token_hs_topk, dim=1) + 1e-8)
                if attention_uniform: # ablation study
                    current_token_attn_topk = torch.ones_like(current_token_attn_topk) / len(current_token_attn_topk)
                if hidden_uniform: # ablation study
                    w_i = torch.ones_like(w_i) / len(w_i)
                icr_score = js_divergence(w_i, current_token_attn_topk)
                icr_scores_layer.append(icr_score)
            top_p_list.append(top_p_layer)

            icr_scores_item.append(icr_scores_layer)
        top_p_mean = np.mean(top_p_list)
        with torch.cuda.device(self.icr_device):
            torch.cuda.empty_cache()
        return icr_scores_item, top_p_mean


def kl_divergence(P, Q):
    """
    计算KL散度 (Kullback-Leibler divergence)。
    
    Args:
        P: 第一个概率分布
        Q: 第二个概率分布
    
    Returns:
        两个分布间的KL散度
    """
    kl_divergence = (P * (P / Q).log()).sum()
    return kl_divergence.item()

def js_divergence(p, q):
    """
    计算JS散度 (Jensen-Shannon divergence)。
    
    JS散度是对称的，且值域在[0,1]之间，常用于衡量两个概率分布的相似度。
    
    Args:
        p: 第一个概率分布
        q: 第二个概率分布
    
    Returns:
        两个分布间的JS散度
    """
    # standardize: p, q -> N(0, 1)
    p = (p - p.mean()) / max(p.std(), 1e-8)
    q = (q - q.mean()) / max(q.std(), 1e-8)
    # softmax: p, q -> [0, 1]
    p = F.softmax(p, dim=0)
    q = F.softmax(q, dim=0)

    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
