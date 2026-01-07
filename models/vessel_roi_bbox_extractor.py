
from omegaconf import DictConfig
import pydoc

import torch


class VesselROIBBoxExtractor(torch.nn.Module):
    def __init__(self
                 , depth_size: int
                 , backbone: DictConfig):
        super(VesselROIBBoxExtractor, self).__init__()

        backbone_type = pydoc.locate(backbone.target_type)
        self.backbone = backbone_type(**backbone.kwargs)

        num_features = self.backbone.num_features

        self.head = torch.nn.Linear(num_features * depth_size // backbone.kwargs.in_chans, 4)
        
        gain = torch.nn.init.calculate_gain('linear')
        torch.nn.init.xavier_uniform_(self.head.weight, gain)
        self.head.bias.data.fill_(0)
    
    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        if 4 == len(volumes.shape):
            volumes.unsqueeze(0)
        
        batch_size, num_groups, channels, height, width = volumes.shape

        volumes = volumes.reshape(batch_size * num_groups, channels, height, width)
        features = self.backbone(volumes)

        features = features.mean(1)

        features = features.reshape(batch_size, num_groups, -1)
        features = features.flatten(1, 2)

        logits = self.head(features)

        logits = logits.sigmoid()

        return logits
    
    @torch.no_grad()
    def inference(self, volumes: torch.Tensor) -> torch.Tensor:
        if 4 == len(volumes.shape):
            volumes = volumes.unsqueeze(0)
        
        predict = self.forward(volumes)
        
        return predict
