import torch.nn as nn
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.vision_transformer import vit_tiny, vit_small, vit_base, vit_large

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'vit_tiny' : [vit_tiny, 192],
    'vit_samll' : [vit_small, 384],
    'vit_base' : [vit_base, 768],
    'vit_large' : [vit_large, 1024],
}

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, arch='resnet50', num_classes=1000, feat_dim=768):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[arch]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class MyModel(nn.Module):
    """backbone + classifier"""
    def __init__(self, arch='resnet50', num_classes=1000):
        super(MyModel, self).__init__()
        model_fun, dim_in = model_dict[arch]
        self.backbon = model_fun()
        self.classifier = LinearClassifier(arch, num_classes, dim_in)

    def forward(self, x):
        feat = self.backbon(x)
        pred = self.classifier(feat)

        return pred







def get_cls_model(config, num_classes=1000):
    return MyModel(config.ARCH, num_classes=num_classes)




