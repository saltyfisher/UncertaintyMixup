import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class ResNet18WithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5):
        super(ResNet18WithDropout, self).__init__()
        # 加载预训练的ResNet18模型
        resnet18 = models.resnet18(pretrained=True)
        
        # 复制所有层，但在每个block之后添加dropout
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        
        # 在每个block之后添加dropout
        self.layer1 = resnet18.layer1
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.layer2 = resnet18.layer2
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.layer3 = resnet18.layer3
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        self.layer4 = resnet18.layer4
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        self.avgpool = resnet18.avgpool
        
        # 在全连接层之前添加dropout
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        # x = self.dropout1(x)
        
        x = self.layer2(x)
        # x = self.dropout2(x)
        
        x = self.layer3(x)
        # x = self.dropout3(x)
        
        x = self.layer4(x)
        x = self.dropout4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 在全连接层之前应用dropout
        # x = self.dropout_fc(x)
        x = self.fc(x)
        
        return x
    
    def enable_dropout(self):
        """启用模型中的所有dropout层"""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
    
    def disable_dropout(self):
        """禁用模型中的所有dropout层"""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()


class ResNet18WithoutDropout(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18WithoutDropout, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        # 确保分类器输出指定数量的类
        if self.resnet18.fc.out_features != num_classes:
            in_features = self.resnet18.fc.in_features
            self.resnet18.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet18(x)


def get_model(model_type='without_dropout'):
    """
    获取指定类型的模型
    Args:
        model_type: 'with_dropout' 或 'without_dropout'
    """
    if model_type == 'with_dropout':
        return ResNet18WithDropout()
    else:
        return ResNet18WithoutDropout()


def get_target_layer(model, model_type='without_dropout'):
    """
    获取目标层用于Grad-CAM
    """
    if model_type == 'with_dropout':
        return model.layer4[-1]
    else:
        return model.resnet18.layer4[-1]