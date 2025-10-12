
import torch

from models.backbone import build_backbone


class DiseaseDetector(torch.nn.Module):
    def __init__(self, backbone_name: str, in_channels: int, num_classes: int):
        super(DiseaseDetector, self).__init__()

        # ResNet base
        self.classifier = build_backbone(backbone_name, in_channels, num_classes)

        if self.classifier is None:
            raise ModuleNotFoundError(name=backbone_name)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(image)
    
    @torch.no_grad()
    def inference(self, image: torch.Tensor) -> torch.Tensor:
        if 3 == len(image.shape):
            image = image.unsqueeze(0)
        
        predict = self.forward(image)
        
        return predict.sigmoid()
