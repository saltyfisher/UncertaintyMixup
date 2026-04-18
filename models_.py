import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class ResNet50WithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5, pretrain=False):
        super(ResNet50WithDropout, self).__init__()
        # 加载预训练的ResNet50模型
        resnet50 = models.resnet50(weights=pretrain)
        
        # 复制所有层，但在每个block之后添加dropout
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        
        # 在每个block之后添加dropout
        self.layer1 = resnet50.layer1
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.layer2 = resnet50.layer2
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.layer3 = resnet50.layer3
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        self.layer4 = resnet50.layer4
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        self.avgpool = resnet50.avgpool
        
        # 在全连接层之前添加dropout
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        x = self.dropout3(x)
        
        x = self.layer4(x)
        x = self.dropout4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 在全连接层之前应用dropout
        x = self.dropout_fc(x)
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


class ResNet50WithoutDropout(nn.Module):
    def __init__(self, num_classes=1000, pretrain=True):
        super(ResNet50WithoutDropout, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrain)
        # 确保分类器输出指定数量的类
        if self.resnet50.fc.out_features != num_classes:
            in_features = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet50(x)


class ResNet18WithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5, pretrain=False):
        super(ResNet18WithDropout, self).__init__()
        # 加载预训练的ResNet18模型
        resnet18 = models.resnet18(weights=pretrain)
        
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
    def __init__(self, num_classes=1000, pretrain=True):
        super(ResNet18WithoutDropout, self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrain)
        # 确保分类器输出指定数量的类
        if self.resnet18.fc.out_features != num_classes:
            in_features = self.resnet18.fc.in_features
            self.resnet18.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet18(x)


class ResNet34WithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5, pretrain=False):
        super(ResNet34WithDropout, self).__init__()
        # 加载预训练的ResNet34模型
        resnet34 = models.resnet34(weights=pretrain)
        
        # 复制所有层，但在每个block之后添加dropout
        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        
        # 在每个block之后添加dropout
        self.layer1 = resnet34.layer1
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.layer2 = resnet34.layer2
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.layer3 = resnet34.layer3
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        self.layer4 = resnet34.layer4
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        self.avgpool = resnet34.avgpool
        
        # 在全连接层之前添加dropout
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(resnet34.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        x = self.dropout3(x)
        
        x = self.layer4(x)
        x = self.dropout4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 在全连接层之前应用dropout
        x = self.dropout_fc(x)
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


class ResNet34WithoutDropout(nn.Module):
    def __init__(self, num_classes=1000, pretrain=True):
        super(ResNet34WithoutDropout, self).__init__()
        self.resnet34 = models.resnet34(pretrained=pretrain)
        # 确保分类器输出指定数量的类
        if self.resnet34.fc.out_features != num_classes:
            in_features = self.resnet34.fc.in_features
            self.resnet34.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet34(x)


class InceptionV3WithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5, pretrain=False):
        super(InceptionV3WithDropout, self).__init__()
        # 加载预训练的InceptionV3模型
        inception = models.inception_v3(weights=pretrain, aux_logits=False)
        
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        
        self.avgpool = inception.avgpool
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # 如果输入尺寸小于75，需要调整
        if x.size()[2] < 75:
            raise ValueError("Input size must be at least 75x75 for Inception v3")
            
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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


class InceptionV3WithoutDropout(nn.Module):
    def __init__(self, num_classes=1000, pretrain=True):
        super(InceptionV3WithoutDropout, self).__init__()
        self.inception = models.inception_v3(pretrained=pretrain, aux_logits=False)
        # 确保分类器输出指定数量的类
        if self.inception.fc.out_features != num_classes:
            in_features = self.inception.fc.in_features
            self.inception.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.inception(x)


class WideResNetWithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5, pretrain=False):
        super(WideResNetWithDropout, self).__init__()
        # 加载预训练的WideResNet模型 (WideResNet-50-2)
        wide_resnet = models.wide_resnet50_2(weights=pretrain)
        
        self.conv1 = wide_resnet.conv1
        self.bn1 = wide_resnet.bn1
        self.relu = wide_resnet.relu
        self.maxpool = wide_resnet.maxpool
        
        self.layer1 = wide_resnet.layer1
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.layer2 = wide_resnet.layer2
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.layer3 = wide_resnet.layer3
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        self.layer4 = wide_resnet.layer4
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        self.avgpool = wide_resnet.avgpool
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(wide_resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        x = self.dropout3(x)
        
        x = self.layer4(x)
        x = self.dropout4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
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


class WideResNetWithoutDropout(nn.Module):
    def __init__(self, num_classes=1000, pretrain=True):
        super(WideResNetWithoutDropout, self).__init__()
        self.wide_resnet = models.wide_resnet50_2(pretrained=pretrain)
        # 确保分类器输出指定数量的类
        if self.wide_resnet.fc.out_features != num_classes:
            in_features = self.wide_resnet.fc.in_features
            self.wide_resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.wide_resnet(x)


class EfficientNetWithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5, pretrain=False, model_version='b0'):
        super(EfficientNetWithDropout, self).__init__()
        # 加载指定版本的EfficientNet模型
        if model_version == 'b0':
            efficientnet = models.efficientnet_b0(weights=pretrain)
        elif model_version == 'b1':
            efficientnet = models.efficientnet_b1(weights=pretrain)
        elif model_version == 'b2':
            efficientnet = models.efficientnet_b2(weights=pretrain)
        elif model_version == 'b3':
            efficientnet = models.efficientnet_b3(weights=pretrain)
        elif model_version == 'b4':
            efficientnet = models.efficientnet_b4(weights=pretrain)
        elif model_version == 'b5':
            efficientnet = models.efficientnet_b5(weights=pretrain)
        elif model_version == 'b6':
            efficientnet = models.efficientnet_b6(weights=pretrain)
        elif model_version == 'b7':
            efficientnet = models.efficientnet_b7(weights=pretrain)
        else:
            efficientnet = models.efficientnet_b0(weights=pretrain)
        
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.classifier = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
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


class EfficientNetWithoutDropout(nn.Module):
    def __init__(self, num_classes=1000, pretrain=True, model_version='b0'):
        super(EfficientNetWithoutDropout, self).__init__()
        # 加载指定版本的EfficientNet模型
        if model_version == 'b0':
            self.efficientnet = models.efficientnet_b0(pretrained=pretrain)
        elif model_version == 'b1':
            self.efficientnet = models.efficientnet_b1(pretrained=pretrain)
        elif model_version == 'b2':
            self.efficientnet = models.efficientnet_b2(pretrained=pretrain)
        elif model_version == 'b3':
            self.efficientnet = models.efficientnet_b3(pretrained=pretrain)
        elif model_version == 'b4':
            self.efficientnet = models.efficientnet_b4(pretrained=pretrain)
        elif model_version == 'b5':
            self.efficientnet = models.efficientnet_b5(pretrained=pretrain)
        elif model_version == 'b6':
            self.efficientnet = models.efficientnet_b6(pretrained=pretrain)
        elif model_version == 'b7':
            self.efficientnet = models.efficientnet_b7(pretrained=pretrain)
        else:
            self.efficientnet = models.efficientnet_b0(pretrained=pretrain)
        
        # 确保分类器输出指定数量的类
        if self.efficientnet.classifier[1].out_features != num_classes:
            in_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)


class EfficientNetV2WithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.5, pretrain=False, model_version='s'):
        super(EfficientNetV2WithDropout, self).__init__()
        # 加载指定版本的EfficientNetV2模型
        if model_version == 's':
            efficientnet = models.efficientnet_v2_s(weights=pretrain)
        elif model_version == 'm':
            efficientnet = models.efficientnet_v2_m(weights=pretrain)
        elif model_version == 'l':
            efficientnet = models.efficientnet_v2_l(weights=pretrain)
        else:
            efficientnet = models.efficientnet_v2_s(weights=pretrain)
        
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.classifier = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
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


class EfficientNetV2WithoutDropout(nn.Module):
    def __init__(self, num_classes=1000, pretrain=True, model_version='s'):
        super(EfficientNetV2WithoutDropout, self).__init__()
        # 加载指定版本的EfficientNetV2模型
        if model_version == 's':
            self.efficientnet = models.efficientnet_v2_s(pretrained=pretrain)
        elif model_version == 'm':
            self.efficientnet = models.efficientnet_v2_m(pretrained=pretrain)
        elif model_version == 'l':
            self.efficientnet = models.efficientnet_v2_l(pretrained=pretrain)
        else:
            self.efficientnet = models.efficientnet_v2_s(pretrained=pretrain)
        
        # 确保分类器输出指定数量的类
        if self.efficientnet.classifier[1].out_features != num_classes:
            in_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)


def get_model(model_type='without_dropout', pretrain=False, model_arch='resnet18', num_classes=2):
    """
    获取指定类型的模型
    Args:
        model_type: 'with_dropout' 或 'without_dropout'
        pretrain: 是否使用预训练权重
        model_arch: 'resnet18' 或 'resnet50'
    """
    if model_arch == 'resnet50':
        if model_type == 'with_dropout':
            return ResNet50WithDropout(num_classes=num_classes, pretrain=pretrain)
        else:
            return ResNet50WithoutDropout(num_classes=num_classes,pretrain=pretrain)
    elif model_arch == 'resnet34':
        if model_type == 'with_dropout':
            return ResNet34WithDropout(num_classes=num_classes,pretrain=pretrain)
        else:
            return ResNet34WithoutDropout(num_classes=num_classes,pretrain=pretrain)
    elif model_arch == 'resnet18':
        if model_type == 'with_dropout':
            return ResNet18WithDropout(num_classes=num_classes,pretrain=pretrain)
        else:
            return ResNet18WithoutDropout(num_classes=num_classes,pretrain=pretrain)
    elif model_arch == 'inception':
        if model_type == 'with_dropout':
            return InceptionV3WithDropout(num_classes=num_classes,pretrain=pretrain)
        else:
            return InceptionV3WithoutDropout(num_classes=num_classes,pretrain=pretrain)
    elif model_arch == 'wideresnet':
        if model_type == 'with_dropout':
            return WideResNetWithDropout(num_classes=num_classes,pretrain=pretrain)
        else:
            return WideResNetWithoutDropout(num_classes=num_classes,pretrain=pretrain)
    elif model_arch.startswith('efficient'):
        # 解析efficientnet_v2版本 (例如: efficientnet_v2_s)
        version = model_arch.split('efficient')[1]
        if model_type == 'with_dropout':
            return EfficientNetV2WithDropout(num_classes=num_classes,pretrain=pretrain, model_version=version)
        else:
            return EfficientNetV2WithoutDropout(num_classes=num_classes,pretrain=pretrain, model_version=version)
    else:  # 默认为resnet18
        if model_type == 'with_dropout':
            return ResNet18WithDropout(num_classes=num_classes,pretrain=pretrain)
        else:
            return ResNet18WithoutDropout(num_classes=num_classes,pretrain=pretrain)
        
def get_target_layer(model, model_type='without_dropout', model_arch='resnet18'):
    """
    获取目标层用于Grad-CAM
    Args:
        model: 模型实例
        model_type: 'with_dropout' 或 'without_dropout'
        model_arch: 'resnet18' 或 'resnet50'
    """
    if model_type == 'with_dropout':
        if model_arch == 'resnet50':
            return model.layer4[-1]
        elif model_arch == 'resnet34':
            return model.layer4[-1]
        elif model_arch == 'resnet18':
            return model.layer4[-1]
        elif model_arch == 'inceptionv3':
            return model.Mixed_7c
        elif model_arch == 'wideresnet':
            return model.layer4[-1]
        elif model_arch.startswith('efficientnet_v2'):
            return model.features[-1]
        elif model_arch.startswith('efficientnet'):
            return model.features[-1]
        else:
            return model.layer4[-1]
    else:
        if model_arch == 'resnet50':
            return model.layer4[-1]
        elif model_arch == 'resnet34':
            return model.layer4[-1]
        elif model_arch == 'resnet18':
            return model.layer4[-1]
        elif model_arch == 'inceptionv3':
            return model.Mixed_7c
        elif model_arch == 'wideresnet':
            return model.wide_resnet.layer4[-1]
        elif model_arch.startswith('efficientnet_v2'):
            # 对于EfficientNetV2，返回features的最后一层
            return model.efficientnet.features[-1]
        elif model_arch.startswith('efficientnet'):
            # 对于EfficientNet，返回features的最后一层
            return model.efficientnet.features[-1]
        else:  # 默认为resnet18
            return model.resnet18.layer4[-1]