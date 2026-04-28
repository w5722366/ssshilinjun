# """
# Implementation of Prof-of-Concept Network: StarNet.

# We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
#     - like NO layer-scale in network design,
#     - and NO EMA during training,
#     - which would improve the performance further.

# Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
# Modified Date: Mar/29/2024
# """
# import torch
# import torch.nn as nn
# from timm.models.layers import DropPath, trunc_normal_
# # from mmdet.registry import MODELS
# from mmselfsup.registry import MODELS

# __all__ = ['starnet_s050', 'starnet_s100', 'starnet_s150', 'starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4']

# model_urls = {
#     "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
#     "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
#     "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
#     "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
# }


# class ConvBN(torch.nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
#         super().__init__()
#         self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
#         if with_bn:
#             self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
#             torch.nn.init.constant_(self.bn.weight, 1)
#             torch.nn.init.constant_(self.bn.bias, 0)


# class Block(nn.Module):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0.):
#         super().__init__()
#         self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
#         self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         self.act = nn.ReLU6()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x1, x2 = self.f1(x), self.f2(x)
#         x = self.act(x1) * x2
#         x = self.dwconv2(self.g(x))
#         x = input + self.drop_path(x)
#         return x

# @MODELS.register_module()
# class StarNet(nn.Module):
#     def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
#         super().__init__()
#         self.num_classes = num_classes
#         self.in_channel = 32
#         # stem layer
#         self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
#         # build stages
#         self.stages = nn.ModuleList()
#         cur = 0
#         for i_layer in range(len(depths)):
#             embed_dim = base_dim * 2 ** i_layer
#             down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
#             self.in_channel = embed_dim
#             blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
#             cur += depths[i_layer]
#             self.stages.append(nn.Sequential(down_sampler, *blocks))
        
#         self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
#         print("Channel sizes: ", self.channel)
#         self.apply(self._init_weights)
#         print("StarNet registered")  # 确认此行是否输出

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear or nn.Conv2d):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         features = []
#         x = self.stem(x)
#         features.append(x)
#         for stage in self.stages:
#             x = stage(x)
#             features.append(x)
#         return features



# def starnet_s1(pretrained=False, **kwargs):
#     model = StarNet(24, [2, 2, 8, 3], **kwargs)
#     if pretrained:
#         url = model_urls['starnet_s1']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"], strict=False)
#     return model



# def starnet_s2(pretrained=False, **kwargs):
#     model = StarNet(32, [1, 2, 6, 2], **kwargs)
#     if pretrained:
#         url = model_urls['starnet_s2']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"], strict=False)
#     return model



# def starnet_s3(pretrained=False, **kwargs):
#     model = StarNet(32, [2, 2, 8, 4], **kwargs)
#     if pretrained:
#         url = model_urls['starnet_s3']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"], strict=False)
#     return model



# def starnet_s4(pretrained=False, **kwargs):
#     model = StarNet(32, [3, 3, 12, 5], **kwargs)
#     if pretrained:
#         url = model_urls['starnet_s4']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"], strict=False)
#     return model


# # very small networks #

# def starnet_s050(pretrained=False, **kwargs):
#     return StarNet(16, [1, 1, 3, 1], 3, **kwargs)



# def starnet_s100(pretrained=False, **kwargs):
#     return StarNet(20, [1, 2, 4, 1], 4, **kwargs)



# def starnet_s150(pretrained=False, **kwargs):
#     return StarNet(24, [1, 2, 4, 2], 3, **kwargs)
"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from mmselfsup.registry import MODELS
import math

from mmcv.cnn import ConvModule

__all__ = ['starnet_s050', 'starnet_s100', 'starnet_s150', 'starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4']

__all__ = ['starnet_s050', 'starnet_s100', 'starnet_s150', 'starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4']

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


def geometric_encoding_single(boxes):
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
    w = x2 - x1
    h = y2 - y1
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    # 添加安全的宽度和高度，防止为零
    safe_w = torch.clamp(w, min=1e-3)  # 防止宽度为 0
    safe_h = torch.clamp(h, min=1e-3)  # 防止高度为 0

    # [K,K] 差值计算
    delta_x = center_x - torch.transpose(center_x, 0, 1)
    delta_x = delta_x / safe_w  # 使用安全的宽度进行归一化
    delta_x = torch.log(torch.abs(delta_x).clamp(min=1e-3))  # 确保对数输入不为 0 或负值

    delta_y = center_y - torch.transpose(center_y, 0, 1)
    delta_y = delta_y / safe_h  # 使用安全的高度进行归一化
    delta_y = torch.log(torch.abs(delta_y).clamp(min=1e-3))  # 确保对数输入不为 0 或负值

    # 宽度和高度变化
    delta_w = torch.log(safe_w / torch.transpose(safe_w, 0, 1))  # 使用安全的宽度
    delta_h = torch.log(safe_h / torch.transpose(safe_h, 0, 1))  # 使用安全的高度

    # [K,K,4] 最终输出
    output = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=2)

    return output


def geometric_encoding_batch(boxes, bs):
    '''
    boxes: tensor[bs*roi_num,4]
    bs: batch size
    '''

    boxes = boxes.reshape(bs, -1, 4)
    bbox_encoding_batch = []
    for bs_id in range(0, bs):
        bbox_encoding = geometric_encoding_single(boxes[bs_id])
        bbox_encoding_batch.append(bbox_encoding)
    # [bs,roi_num,roi_num,4]
    output = torch.stack(bbox_encoding_batch, dim=0)

    return output

class RelationModule(nn.Module):
    def __init__(self, box_feat_dim=96, group=16, geo_feat_dim=64):
        super(RelationModule, self).__init__()
        self.box_feat_dim = box_feat_dim
        self.group = group
        self.geo_feat_dim = geo_feat_dim

        self.dim = 128  # 1024
        self.num_heads = group   # 16
        self.head_dim = self.dim // self.num_heads  # 64

        self.tanh = nn.Tanh()
        self.geo_emb_fc = nn.Linear(4, geo_feat_dim)
        self.box_geo_conv = ConvModule(geo_feat_dim, self.num_heads, 1)

        # 定义线性层用于计算 q, k, v
        self.q1 = nn.Linear(self.dim // 2, self.dim // 2, bias=True)
        self.q2 = nn.Linear(self.dim // 2, self.dim // 2, bias=True)
        self.k1 = nn.Linear(self.dim // 2, self.dim // 2, bias=True)
        self.k2 = nn.Linear(self.dim // 2, self.dim // 2, bias=True)
        self.v1 = nn.Linear(self.dim // 2, self.dim // 2, bias=True)
        self.v2 = nn.Linear(self.dim // 2, self.dim // 2, bias=True)

        self.proj1 = nn.Linear(self.dim // 2, self.dim // 2, bias=True)
        self.proj2 = nn.Linear(self.dim // 2, self.dim // 2, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, box_appearance_feat, boxes, bs):
        roi_num = int(box_appearance_feat.size(0) / bs)

        # 几何特征编码
        box_geo_encoded = geometric_encoding_batch(boxes, bs)
        box_geo_feat = self.tanh(self.geo_emb_fc(box_geo_encoded))
        box_geo_feat = box_geo_feat.permute(0, 3, 1, 2)  # [bs, geo_feat_dim, roi_num, roi_num]
        box_geo_feat_wg = self.box_geo_conv(box_geo_feat)  # [bs, num_heads, roi_num, roi_num]
        box_geo_feat_wg = torch.sigmoid(box_geo_feat_wg)  # 确保值在 [0,1] 范围内

        # 调整外观特征的形状
        x = box_appearance_feat.reshape(bs, roi_num, self.dim)
        x = x.reshape(bs, roi_num, 2, self.dim // 2).permute(0, 2, 1, 3).contiguous()
        # x 的形状：[bs, 2, roi_num, 512]

        # 计算 q, k, v
        q_list = []
        k_list = []
        v_list = []
        for i in range(2):
            xi = x[:, i]  # [bs, roi_num, 512]
            qi = xi + getattr(self, f'q{i+1}')(xi)
            ki = xi + getattr(self, f'k{i+1}')(xi)
            vi = xi + getattr(self, f'v{i+1}')(xi)
            q_list.append(qi)
            k_list.append(ki)
            v_list.append(vi)

        q = torch.cat(q_list, dim=-1)  # [bs, roi_num, 1024]
        k = torch.cat(k_list, dim=-1)
        v = torch.cat(v_list, dim=-1)

        # 调整形状以适用于多头注意力
        q = q.reshape(bs, roi_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [bs, num_heads, roi_num, head_dim]
        k = k.reshape(bs, roi_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(bs, roi_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力权重
        # print("q before normalize:", q)
        # print("k before normalize:", k)
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        # print("q after normalize:", q)
        # print("k after normalize:", k)
        attn = (q @ k.transpose(-2, -1))  # [bs, num_heads, roi_num, roi_num]
        # print("attn after matmul:", attn)
        attn = attn / math.sqrt(self.head_dim)  # 缩放
        attn = attn - attn.max(dim=-1, keepdim=True)[0]  # 数值稳定化
        attn = attn + box_geo_feat_wg  # 添加几何特征
        attn = self.softmax(attn)

        # 计算输出
        x = (attn @ v).permute(0, 2, 1, 3).reshape(bs, roi_num, self.dim)
        output = x.reshape(bs * roi_num, self.dim)
        output = output.view(bs, self.dim, 1, 1)  # 调整为四维张量 [bs, self.dim, 1, 1]
        return output
    
@MODELS.register_module()
class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        self.base_dim = base_dim
        self.dim = base_dim * 2 ** 2
        
        # 初始化RelationModule
        self.relation_module = RelationModule()  # RelationModule 被加入到网络
        
        # stem layer 和 stages
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        
        dummy_boxes = torch.randn(1, 4)
        
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640), boxes=dummy_boxes)]
#         self.apply(self._init_weights)

    def forward(self, x, boxes=None):
        features = []
        x = self.stem(x)
        features.append(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            
            # 在特定的层后面加入RelationModule来增强特征（例如在中间某层）
            if i == 2 and boxes is not None:# 假设在第二层后加入 RelationModule
                box_appearance_feat = x
                box_appearance_feat = F.adaptive_avg_pool2d(box_appearance_feat, (1, 1))  # [bs, self.dim, 1, 1]
                bs = box_appearance_feat.size(0)  # 批次大小

                print(f"box_appearance_feat shape after pooling: {box_appearance_feat.shape}")

                # 使用 RelationModule
                relation_output = self.relation_module(box_appearance_feat, boxes, bs)  # 输出形状为 [bs, self.dim, 1, 1]

                # 将 relation_output 上采样到 x 的大小
                relation_output = F.interpolate(relation_output, size=x.shape[2:], mode='bilinear', align_corners=False)

                # 相加时通道数匹配
                x = x + relation_output         
        return features  # 返回整个网络的特征
    
def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model



def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model



def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model



def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


# very small networks #

def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)



def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)



def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)


