import torch.nn as nn
import torch
class EmbFcBlock(nn.Module):
    def __init__(self):
        super(EmbFcBlock, self).__init__()
        self.nn1 = nn.Linear(in_features=512, out_features=512)
        self.norm = nn.InstanceNorm2d(num_features=512)
        self.nn2 = nn.Linear(in_features=512, out_features=512)
        self.nn3 = nn.Linear(in_features=512, out_features=512)

    def forward(self, ten):
        t = self.nn1(ten)
        t = torch.norm(t)
        t = self.norm(t)
        t = torch.nn.functional.relu(t)

        t = self.nn2(t)
        t = torch.norm(t)
        t = self.norm(t)
        t = torch.nn.functional.relu(t)
        
        t = self.nn3(t)
        t = torch.norm(t)
        t = self.norm(t)
        t = torch.nn.functional.relu(t)

        return t
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def addStyle(target, source):
    t = adaptive_instance_normalization(target, source)
    t = 0.75 * t + (1 - 0.75) * source
    return t

if __name__ == '__main__':
    print('hello')
