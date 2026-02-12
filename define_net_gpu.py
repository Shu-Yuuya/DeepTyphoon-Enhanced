import torch
import torch.nn as nn
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #加载预训练的模型
        self.resnet = models.resnet18(pretrained=True)

        #手动迁移权重
        with torch.no_grad():
            old_conv1 = self.resnet.conv1
            new_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv1.weight.copy_(old_conv1.weight[:, :2, :, :])
            self.resnet.conv1 = new_conv1

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.resnet(x)