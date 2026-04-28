# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class SimSiam(BaseModel):
    """SimSiam.

    Implementation of `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_. The operation of fixing learning rate
    of predictor is in `engine/hooks/simsiam_hook.py`.
    """

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        return self.backbone(inputs[0])
    def loss(self, inputs: List[torch.Tensor],
         data_samples: List[SelfSupDataSample],
         **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training with added noise.
        
        Args:
        inputs (List[torch.Tensor]): The input images.
        data_samples (List[SelfSupDataSample]): All elements required
            during the forward function.
            Returns:
            Dict[str, Tensor]: A dictionary of loss components. """
        img_v1 = inputs[0]
        img_v2 = inputs[1]

        # Add Gaussian noise to img_v1 and img_v2
        noise_std = 0.01  # Adjust the noise standard deviation
        noise_v1 = torch.randn_like(img_v1) * noise_std
        noise_v2 = torch.randn_like(img_v2) * noise_std

        img_v1_noisy = img_v1 + noise_v1
        img_v2_noisy = img_v2 + noise_v2

        avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        z1 = self.neck([avg_pool(self.backbone(img_v1_noisy)[-1])])[0]  # Added pooling
        z2 = self.neck([avg_pool(self.backbone(img_v2_noisy)[-1])])[0]

        loss_1 = self.head(z1, z2)
        loss_2 = self.head(z2, z1)

        losses = dict(loss=0.5 * (loss_1 + loss_2))
        return losses


#     def loss(self, inputs: List[torch.Tensor],
#              data_samples: List[SelfSupDataSample],
#              **kwargs) -> Dict[str, torch.Tensor]:
#         """The forward function in training.

#         Args:
#             inputs (List[torch.Tensor]): The input images.
#             data_samples (List[SelfSupDataSample]): All elements required
#                 during the forward function.

#         Returns:
#             Dict[str, Tensor]: A dictionary of loss components.
#         """
#         img_v1 = inputs[0]
#         img_v2 = inputs[1]
# #         print('img_v2:',img_v2)

# #         z1 = self.neck(self.backbone(img_v1))[0]  # NxC
# #         z2 = self.neck(self.backbone(img_v2))[0]  # NxC
#         avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
#         z1 = self.neck([avg_pool(self.backbone(img_v1)[-1])])[0]  # 添加池化
#         z2 = self.neck([avg_pool(self.backbone(img_v2)[-1])])[0]
        

#         loss_1 = self.head(z1, z2)
#         loss_2 = self.head(z2, z1)

#         losses = dict(loss=0.5 * (loss_1 + loss_2))
#         return losses
