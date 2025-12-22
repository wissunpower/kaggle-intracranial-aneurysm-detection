
from omegaconf import DictConfig
import pydoc

import torch


class VesselSegmentExtractor(torch.nn.Module):
    def __init__(self
                 , backbone: DictConfig):
        super(VesselSegmentExtractor, self).__init__()

        backbone_type = pydoc.locate(backbone.target_type)
        backbone_kwargs = dict()
        for key, value in backbone.kwargs.items():
            if isinstance(value, DictConfig) and 'target_type' in value:
                new_value = pydoc.locate(value['target_type'])
            else:
                new_value = value
            backbone_kwargs[key] = new_value
        
        self.backbone = backbone_type(**backbone_kwargs)
    
    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        if 4 == len(volumes.shape):
            volumes.unsqueeze(0)

        volumes = self.backbone(volumes)

        return volumes[0]
    
    @torch.no_grad()
    def inference(self, volumes: torch.Tensor) -> torch.Tensor:
        if 4 == len(volumes.shape):
            volumes = volumes.unsqueeze(0)
        
        predict = self.forward(volumes)
        
        return predict.sigmoid()
