
import torch
import torchvision.models as models


def build_resnet(resnet_name: str, in_channels: int, num_classes: int) -> torch.nn.Module|None:

    resnet = None
    
    # backbone_name is 'resnet18'
    if models.resnet18.__name__.lower() == resnet_name:
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # backbone_name is 'resnet34'
    elif models.resnet34.__name__.lower() == resnet_name:
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # backbone_name is 'resnet50'
    elif models.resnet50.__name__.lower() == resnet_name:
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # backbone_name is 'resnet101'
    elif models.resnet101.__name__.lower() == resnet_name:
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    # backbone_name is 'resnet152'
    elif models.resnet152.__name__.lower() == resnet_name:
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    
    if resnet is not None:
        conv1 = resnet.conv1
        resnet.conv1 = torch.nn.Conv2d(in_channels, conv1.out_channels
                                       , conv1.kernel_size[0], conv1.stride[0], conv1.padding[0], bias=False)
        torch.nn.init.kaiming_uniform_(resnet.conv1.weight)

        fc_in_features = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(fc_in_features, num_classes)
        
        gain = torch.nn.init.calculate_gain('linear')
        torch.nn.init.xavier_uniform_(resnet.fc.weight, gain)
        resnet.fc.bias.data.fill_(0)

    return resnet

def build_mobilenet(mobilenet_name: str, in_channels: int, num_classes: int) -> torch.nn.Module|None:

    mobilenet = None
    classifier_index = int(1)
    
    # backbone_name is 'mobilenet_v2'
    if models.mobilenet_v2.__name__.lower() == mobilenet_name:
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        classifier_index = int(1)

    # backbone_name is 'mobilenet_v3_small'
    elif models.mobilenet_v3_small.__name__.lower() == mobilenet_name:
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        classifier_index = int(3)

    # backbone_name is 'mobilenet_v3_large'
    elif models.mobilenet_v3_large.__name__.lower() == mobilenet_name:
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        classifier_index = int(3)

    if mobilenet is not None:
        conv1 = mobilenet.features[0][0]
        mobilenet.features[0][0] = torch.nn.Conv2d(in_channels, conv1.out_channels
                                       , conv1.kernel_size[0], conv1.stride[0], conv1.padding[0], bias=False)
        torch.nn.init.kaiming_uniform_(mobilenet.features[0][0].weight)

        last_fc_in_features = mobilenet.classifier[classifier_index].in_features
        mobilenet.classifier[classifier_index] = torch.nn.Linear(last_fc_in_features, num_classes)

        gain = torch.nn.init.calculate_gain('linear')
        torch.nn.init.xavier_uniform_(mobilenet.classifier[classifier_index].weight, gain)
        mobilenet.classifier[classifier_index].bias.data.fill_(0)

    return mobilenet

def build_efficientnet(efficientnet_name: str, in_channels: int, num_classes: int) -> torch.nn.Module|None:

    efficientnet = None
    
    # backbone_name is 'efficientnet_b0'
    if models.efficientnet_b0.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # backbone_name is 'efficientnet_b1'
    elif models.efficientnet_b1.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)

    # backbone_name is 'efficientnet_b2'
    elif models.efficientnet_b2.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

    # backbone_name is 'efficientnet_b3'
    elif models.efficientnet_b3.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    # backbone_name is 'efficientnet_b4'
    elif models.efficientnet_b4.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

    # backbone_name is 'efficientnet_b5'
    elif models.efficientnet_b5.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)

    # backbone_name is 'efficientnet_b6'
    elif models.efficientnet_b6.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT)

    # backbone_name is 'efficientnet_v2_s'
    elif models.efficientnet_v2_s.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # backbone_name is 'efficientnet_v2_m'
    elif models.efficientnet_v2_m.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

    # backbone_name is 'efficientnet_v2_l'
    elif models.efficientnet_v2_l.__name__.lower() == efficientnet_name:
        efficientnet = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
    
    if efficientnet is not None:
        conv1 = efficientnet.features[0][0]
        efficientnet.features[0][0] = torch.nn.Conv2d(in_channels, conv1.out_channels
                                       , conv1.kernel_size[0], conv1.stride[0], conv1.padding[0], bias=False)
        torch.nn.init.kaiming_uniform_(efficientnet.features[0][0].weight)

        last_fc_in_features = efficientnet.classifier[1].in_features
        efficientnet.classifier[1] = torch.nn.Linear(last_fc_in_features, num_classes)

        gain = torch.nn.init.calculate_gain('linear')
        torch.nn.init.xavier_uniform_(efficientnet.classifier[1].weight, gain)
        efficientnet.classifier[1].bias.data.fill_(0)

    return efficientnet

def build_swin(swin_name: str, in_channels: int, num_classes: int) -> torch.nn.Module|None:

    swinnet = None
    
    # backbone_name is 'swin_t'
    if models.swin_t.__name__.lower() == swin_name:
        swinnet = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

    # backbone_name is 'swin_s'
    elif models.swin_s.__name__.lower() == swin_name:
        swinnet = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)

    # backbone_name is 'swin_b'
    elif models.swin_b.__name__.lower() == swin_name:
        swinnet = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)

    # backbone_name is 'swin_v2_t'
    elif models.swin_v2_t.__name__.lower() == swin_name:
        swinnet = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)

    # backbone_name is 'swin_v2_s'
    elif models.swin_v2_s.__name__.lower() == swin_name:
        swinnet = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)

    # backbone_name is 'swin_v2_b'
    elif models.swin_v2_b.__name__.lower() == swin_name:
        swinnet = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
    
    if swinnet is not None:
        conv1 = swinnet.features[0][0]
        swinnet.features[0][0] = torch.nn.Conv2d(in_channels, conv1.out_channels
                                       , conv1.kernel_size[0], conv1.stride[0])
        torch.nn.init.kaiming_uniform_(swinnet.features[0][0].weight)

        last_fc_in_features = swinnet.head.in_features
        swinnet.head = torch.nn.Linear(last_fc_in_features, num_classes)

        gain = torch.nn.init.calculate_gain('linear')
        torch.nn.init.xavier_uniform_(swinnet.head.weight, gain)
        swinnet.head.bias.data.fill_(0)

    return swinnet

def build_backbone(backbone_name: str, in_channels: int, num_classes: int) -> torch.nn.Module|None:

    backbone = None
    
    # backbone_name is 'resnet*'
    if backbone_name.startswith(models.ResNet.__name__.lower()):
        # ResNet base
        backbone = build_resnet(backbone_name, in_channels, num_classes)

    # backbone_name is 'mobilenet*'
    elif backbone_name.startswith('mobilenet'):
    # elif backbone_name.startswith(models.MobileNetV2.__name__.lower()) \
    #     or backbone_name.startswith(models.MobileNetV3.__name__.lower()):
        # MobileNet base
        backbone = build_mobilenet(backbone_name, in_channels, num_classes)

    # backbone_name is 'efficientnet*'
    elif backbone_name.startswith(models.EfficientNet.__name__.lower()):
        # EfficientNet base
        backbone = build_efficientnet(backbone_name, in_channels, num_classes)

    # backbone_name is 'swin*'
    elif backbone_name.startswith('swin'):
    # elif backbone_name.startswith(models.SwinTransformer.__name__.lower()):
        # Swin_V2_S base
        backbone = build_swin(backbone_name, in_channels, num_classes)

    return backbone
