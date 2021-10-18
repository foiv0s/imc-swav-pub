from torch import nn
import torch
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

'''Modify version of Pytorch's ResNet'''


class ResNet(nn.Module):
    def __init__(self, block, layers, conv, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.covn1 = conv
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])  # 16x16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc.init_weights(1.)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        self.layer_list = nn.ModuleList([self.covn1, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool])
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        layer_acts = [x]
        for i, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)
        return layer_acts[-1]

    def forward(self, x):
        return self._forward_impl(x)


class Projection(nn.Module):
    def __init__(self, n_input, n_out=128):
        super(Projection, self).__init__()
        # self.project = nn.Sequential(nn.Conv2d(n_input, n_input, 1, 1, 0),
        #                             # nn.BatchNorm2d(n_input, affine=True),
        #                             # nn.ReLU(inplace=True),
        #                             nn.Conv2d(n_input, n_out, 1, 1, 0, bias=True))
        self.project = nn.Sequential(nn.Linear(n_input, n_input, bias=True),
                                     # nn.BatchNorm1d(n_input, affine=True),
                                     # nn.ReLU(inplace=True),
                                     nn.Linear(n_input, n_out, bias=True))
        return

    def forward(self, r1_x):
        # out = self.project(r1_x)
        # out = nn.functional.normalize(out, dim=1, p=2)
        return self.project(r1_x)


class Prototypes(nn.Module):
    def __init__(self, n_input, n_out=1000):
        super(Prototypes, self).__init__()
        # self.prototypes = nn.Conv2d(n_input, n_out, 1, 1, 0, bias=False)
        self.prototypes = nn.Linear(n_input, n_out, bias=False)
        return

    def forward(self, r1_x):
        r1_x = nn.functional.normalize(r1_x, dim=1, p=2)
        # if len(r1_x.shape) != 4:
        #    r1_x = r1_x.unsqueeze(-1).unsqueeze(-1)
        return self.prototypes(r1_x)  # .squeeze(-1).squeeze(-1)


class Classifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=1024):
        super(Classifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden

        self.aux_head = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=True),
            nn.BatchNorm1d(n_hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.BatchNorm1d(n_hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_classes, bias=True)
            # nn.Conv2d(n_input, n_hidden, 1, 1, 0),
            # nn.BatchNorm2d(n_hidden, affine=True),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(n_hidden, n_hidden, 1, 1, 0),
            # nn.BatchNorm2d(n_hidden, affine=True),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(n_hidden, n_classes, 1, 1, 0)
        )
        return

    def forward(self, r1_x):
        # Always detach so to train only omega parameters
        r1_x = r1_x.detach()
        px = self.aux_head(r1_x)
        return px.squeeze(-1).squeeze(-1)


class MLPClassifier(nn.Module):
    def __init__(self, n_classes, n_input, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes

        self.block_forward = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(self.n_input, n_classes, bias=True)
        )

    def forward(self, x):
        x = x.detach()
        x = torch.flatten(x, 1)
        logits = self.block_forward(x)
        return logits
