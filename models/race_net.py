import torch
import torch.nn as nn
from torchvision.models import resnet34
from configs.paths_config  import model_paths
from torchvision.transforms import transforms 
import torch.nn.functional as F

"""
    the representation over the out[:7] (first 7 in one hot dim)
    0 = 'White'
    1 = 'Black'
    2 = 'Latino_Hispanic'
    3 = 'East Asian'
    4 = 'Southeast Asian'
    5 = 'Indian'
    6 = 'Middle Eastern'
"""



class Race_net(nn.Module):
    def __init__(self, device):
        super(Race_net,self).__init__()
        self.backbone =  resnet34(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 18)
        self.backbone.load_state_dict(torch.load(model_paths["resnet_race_classifier"], map_location=torch.device("cpu")))
        self.backbone.to(device=device)
        self.backbone.eval()

    def get_c(self, x):
        out = x
        out = self.backbone.conv1(out)

        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)

        out = self.backbone.maxpool(out)

        out = self.backbone.layer1(out)

        out = self.backbone.layer2(out)
        out1=  out
        out = self.backbone.layer3(out)
        out2=  out
        out = self.backbone.layer4(out)
        out3=  out
        return [out1,out2,out3]
    
    def get_features(self, x):
        out = x
        out = self.backbone.conv1(out)
    
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)
        
        out = self.backbone.layer1(out)
        
        out = self.backbone.layer2(out)
        
        out = self.backbone.layer3(out)
        
        out = self.backbone.layer4(out)
        
        out = self.backbone.avgpool(out)
        
        out = out.reshape(out.shape[0], out.shape[1])
        return out
    
    def get_feature_channel(self, x):
        out = x
        out = self.backbone.conv1(out)
        
        out = self.backbone.bn1(out)
        
        out = self.backbone.relu(out)
        
        out = self.backbone.maxpool(out)
        
        out = self.backbone.layer1(out)
        
        out = self.backbone.layer2(out)
        
        out = self.backbone.layer3(out)
        
        out = self.backbone.layer4(out)
        
        out = out.reshape(out.shape[0], int(out.shape[1]/2) , 2*out.shape[2]*out.shape[3])
        return out.unsqueeze(dim=1)
    
    def forward(self, x):
        out = self.backbone(x)
        # print(f"{out.shape=}")
        return out[:,:7]