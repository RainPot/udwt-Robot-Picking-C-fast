from backbones.resnet import resnet10, resnet50, resnet101
from backbones.hourglass import hourglass_net
from backbones.dla import dla34


def get_backbone(backbone, pretrained=False, num_stacks=2):
    if backbone == 'resnet10':
        return resnet10(pretrained=pretrained)
    elif backbone == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        return resnet101(pretrained=pretrained)
    elif backbone == 'hourglass':
        return hourglass_net(num_stacks=num_stacks)
    elif backbone == 'dla':
        return dla34(pretrained=True)
    else:
        return hourglass_net(num_stacks=num_stacks)
