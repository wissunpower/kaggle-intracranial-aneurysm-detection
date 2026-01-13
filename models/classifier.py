
import torch


class DiseaseDetector(torch.nn.Module):
    def __init__(self
                 , input_volume_channels: int
                 , in_channels: int
                 , num_classes: int
                 , backbone
                 ):
        super(DiseaseDetector, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone = backbone

        num_features = self.backbone.num_features
        num_groups = int(input_volume_channels // self.in_channels)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(num_features * num_groups, self.num_classes)
        
        gain = torch.nn.init.calculate_gain('linear')
        torch.nn.init.xavier_uniform_(self.head.weight, gain)
        self.head.bias.data.fill_(0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = images.shape
        num_groups = int(channels // self.in_channels)

        images = images.reshape(batch_size * num_groups, self.in_channels, height, width)

        features = self.backbone(images)

        if len(features.shape) == 3:
            features = features.mean(1)
        elif len(features.shape) > 3:
            features = self.avg_pool(features).flatten(1, 3)
        
        features = features.reshape(batch_size, -1)
        
        logits = self.head(features)

        return logits
    
    @torch.no_grad()
    def inference(self, image: torch.Tensor) -> torch.Tensor:
        if 3 == len(image.shape):
            image = image.unsqueeze(0)
        
        predict = self.forward(image)
        
        return predict.sigmoid()
