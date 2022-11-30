import torch
import torch.nn as nn
import opacus
from opacus.validators import ModuleValidator

class WideResnet50(nn.Module): 
    def __init__(self): 
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        if not ModuleValidator.is_valid(self.model):
            self.model = ModuleValidator.fix(self.model, strict=False)
            
    def forward(self, x, training=False): 
        if training: 
            self.train()
        else: 
            self.eval()
        
        return self.model(x)
  
