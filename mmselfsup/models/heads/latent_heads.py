# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmengine.dist import all_reduce, get_world_size
from mmengine.model import BaseModule
from scipy.stats import wasserstein_distance

from mmselfsup.registry import MODELS
from typing import Tuple


@MODELS.register_module()
class LatentPredictHead(BaseModule):
    """Head for latent feature prediction with Wasserstein distance.

    This head builds a predictor, which can be any registered neck component.
    For example, BYOL and SimSiam call this head and build NonLinearNeck.
    It also implements similarity loss between two forward features.

    Args:
        loss (dict): Config dict for the loss.
        predictor (dict): Config dict for the predictor.
        use_wasserstein (bool): Whether to use Wasserstein distance. Default: True.
        wasserstein_weight (float): Weight for Wasserstein distance in the total loss. Default: 1.0.
    """

    def __init__(self, loss: dict, predictor: dict, 
                 use_wasserstein: bool = True, wasserstein_weight: float = 1.0) -> None:
        super().__init__()
        self.loss = MODELS.build(loss)
        self.predictor = MODELS.build(predictor)
        self.use_wasserstein = use_wasserstein
        self.wasserstein_weight = wasserstein_weight

    def wasserstein_distance(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Wasserstein distance between input and target features."""
        # Use torch.cdist to compute pairwise distances (approximation to Wasserstein distance)
        # The scipy-based version (cpu().numpy()) could be slow, so we use this as an alternative.
        wasserstein_loss = torch.cdist(input, target, p=1).mean()  # Using p=1 for L1 (Manhattan) distance as an approximation.
        return wasserstein_loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Forward head with optional Wasserstein distance.

        Args:
            input (torch.Tensor): NxC input features.
            target (torch.Tensor): NxC target features.

        Returns:
            torch.Tensor: The latent predict loss.
        """
        pred = self.predictor([input])[0]
        target = target.detach()

        # Compute the standard similarity loss (e.g., CosineSimilarityLoss)
        loss = self.loss(pred, target)

        # Optionally add Wasserstein distance to the loss
        if self.use_wasserstein:
            wasserstein_dist = self.wasserstein_distance(pred, target)
            # Combine Wasserstein distance with the existing loss using a weight
            loss += self.wasserstein_weight * wasserstein_dist

        return loss

# @MODELS.register_module()
# class LatentPredictHead(BaseModule):
#     """Head for latent feature prediction.

#     This head builds a predictor, which can be any registered neck component.
#     For example, BYOL and SimSiam call this head and build NonLinearNeck.
#     It also implements similarity loss between two forward features.

#     Args:
#         loss (dict): Config dict for the loss.
#         predictor (dict): Config dict for the predictor.
#     """

#     def __init__(self, loss: dict, predictor: dict) -> None:
#         super().__init__()
#         self.loss = MODELS.build(loss)
#         self.predictor = MODELS.build(predictor)

#     def forward(self, input: torch.Tensor,
#                 target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Forward head.

#         Args:
#             input (torch.Tensor): NxC input features.
#             target (torch.Tensor): NxC target features.

#         Returns:
#             torch.Tensor: The latent predict loss.
#         """
#         pred = self.predictor([input])[0]
#         target = target.detach()

#         loss = self.loss(pred, target)

#         return loss


@MODELS.register_module()
class LatentCrossCorrelationHead(BaseModule):
    """Head for latent feature cross correlation.

    Part of the code is borrowed from `script
    <https://github.com/facebookresearch/barlowtwins/blob/main/main.py>`_.

    Args:
        in_channels (int): Number of input channels.
        loss (dict): Config dict for module of loss functions.
    """

    def __init__(self, in_channels: int, loss: dict) -> None:
        super().__init__()
        self.world_size = get_world_size()
        self.bn = nn.BatchNorm1d(in_channels, affine=False)
        self.loss = MODELS.build(loss)

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Forward head.

        Args:
            input (torch.Tensor): NxC input features.
            target (torch.Tensor): NxC target features.

        Returns:
            torch.Tensor: The cross correlation loss.
        """
        # cross-correlation matrix
        cross_correlation_matrix = self.bn(input).T @ self.bn(target)
        cross_correlation_matrix.div_(input.size(0) * self.world_size)

        all_reduce(cross_correlation_matrix)

        loss = self.loss(cross_correlation_matrix)
        return loss

# @MODELS.register_module()
# class LatentCrossCorrelationHead(BaseModule):
#     """Head for latent feature cross correlation with Wasserstein distance.

#     Args:
#         in_channels (int): Number of input channels.
#         loss (dict): Config dict for module of loss functions.
#         distance_metric (str): Type of distance metric to use ('euclidean', 'cosine', 'wasserstein').
#     """

#     def __init__(self, in_channels: int, loss: dict, distance_metric: str = 'wasserstein') -> None:
#         super().__init__()
#         self.world_size = get_world_size()  # 用于分布式训练时的同步
#         self.bn = nn.BatchNorm1d(in_channels, affine=False)  # 特征归一化
#         self.loss = MODELS.build(loss)  # 构建损失函数
#         self.distance_metric = distance_metric  # 距离度量类型

#     def wasserstein_distance(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """Calculate Wasserstein distance between input and target features."""
#         loss = 0
#         for i in range(input.size(0)):
#             # 逐样本计算 Wasserstein 距离
#             w_distance = wasserstein_distance(input[i].cpu().numpy(), target[i].cpu().numpy())
#             loss += w_distance
#         return torch.tensor(loss / input.size(0), device=input.device)

#     def forward(self, input: torch.Tensor,
#                 target: torch.Tensor) -> torch.Tensor:
#         """Forward head with Wasserstein distance.

#         Args:
#             input (torch.Tensor): NxC input features.
#             target (torch.Tensor): NxC target features.

#         Returns:
#             torch.Tensor: The cross correlation loss.
#         """
#         # Step 1: Normalize the input features (归一化特征)
#         input_norm = self.bn(input)
#         target_norm = self.bn(target)

#         # Step 2: Compute the cross-correlation matrix (计算交叉相关矩阵)
#         cross_correlation_matrix = input_norm.T @ target_norm
#         cross_correlation_matrix.div_(input.size(0) * self.world_size)

#         # Step 3: Adjust cross-correlation matrix with the chosen distance metric
#         distance_matrix = self.wasserstein_distance(input_norm, target_norm)
#         cross_correlation_matrix.mul_(1 / (1 + distance_matrix.unsqueeze(1)))

#         # Step 4: All reduce operation to sync cross-correlation matrix across different GPUs
#         all_reduce(cross_correlation_matrix)

#         # Step 5: Compute loss based on the modified cross-correlation matrix
#         loss = self.loss(cross_correlation_matrix)
#         print('wasserstein succesfully')
#         return loss