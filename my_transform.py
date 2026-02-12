import torch
import torchvision.transforms as transforms

class demension_reduce(object):
    def __call__(self, tensor):
        return tensor[0:2] #2通道逻辑 

# 训练集
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),  
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    demension_reduce(),
    transforms.Normalize((0.5, 0.5), (0.5, 0.5))
])

# 测试集
test_val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    demension_reduce(),
    transforms.Normalize((0.5, 0.5), (0.5, 0.5))
])