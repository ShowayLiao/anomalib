import torch
import torch.nn as nn


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
