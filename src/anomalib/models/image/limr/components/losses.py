import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def each_patch_loss_function(a, b):
    cos_loss = nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_tem = a[item].permute(0, 2, 3, 1)
        b_tem = b[item].permute(0, 2, 3, 1)
        loss += torch.mean(
            1 - cos_loss(
                a_tem.contiguous().view(-1, a_tem.shape[-1]),
                b_tem.contiguous().view(-1, b_tem.shape[-1]),
            )
        )
    return loss


class LiMRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_features, student_features):
        return each_patch_loss_function(teacher_features, student_features)


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([fs_list[0].shape[0], out_size, out_size])
    else:
        anomaly_map = np.zeros([fs_list[0].shape[0], out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map.squeeze(1).cpu().detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list
