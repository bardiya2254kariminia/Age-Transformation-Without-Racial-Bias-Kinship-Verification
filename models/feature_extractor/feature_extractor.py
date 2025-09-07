import torch 
import torch.nn as nn
from torchvision.models import resnet18 , resnet34
class Feature_extractor(nn.Module):
    def __init__(self , backbone_model = 18 , pretrained = True , freeze = True):
        super(Feature_extractor,self).__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.fc = nn.Linear(in_features=1000 ,
                            out_features=512)
        for param in self.backbone.parameters():
          param.requires_grad = False
    def forward(self, x):
        out = self.backbone(x)
        return self.fc(out)