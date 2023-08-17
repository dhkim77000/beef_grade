import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torch
import torchvision.models as models
import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

class EffNet(nn.Module):
    def __init__(self, args, num_classes:int, **kwargs):
        super(EffNet, self).__init__()
        self.model = timm.create_model(args.model_name, pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features = in_features, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output
    
class Inception(nn.Module):
    def __init__(self, args, num_classes:int, **kwargs):
        super(Inception, self).__init__()
        self.model = timm.create_model('inception_v4', pretrained=True)
        in_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Sequential(
            nn.Linear(in_features = in_features, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output
    
class Vit(nn.Module):
    def __init__(self, args, num_classes:int, **kwargs):
        super(Vit, self).__init__()
        self.model = timm.create_model('regnetz_e8', pretrained=True)
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Sequential(
            nn.Linear(in_features = in_features, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output
    
class Dino(nn.Module):
    def __init__(self, args, num_classes:int, **kwargs):
        super(Dino, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        in_features = self.model.blocks[0].mlp.fc1.in_features
        self.model.blocks[0].mlp = nn.Sequential(
            nn.Linear(in_features = 384, out_features=128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=32),
            nn.GELU(),
            nn.Linear(in_features=32, out_features=num_classes)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output
       

class RESNEXT50(nn.Module):
    def __init__(self, args, num_classes, pretrained=False):
        super(RESNEXT50, self).__init__()
  
        self.model = models.resnext50_32x4d(num_classes = num_classes, pretrained= pretrained)

        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=256, bias=True),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=256, out_features=num_classes, bias=True)
        )
        
    def forward(self, x):
        return self.model(x)