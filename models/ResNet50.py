# -*- coding: utf-8 -*-
 
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .BasicModule import BasicModule
 
class residual_block(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,  bias=False, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = stride, kernel_size=3 , padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4,  kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
 
    def forward(self, x):
        residual = x
 
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
 
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu(bn2)
 
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        if self.downsample is not None:
            residual = self.downsample(x)
        bn3 += residual
        out = self.relu(bn3)
 
        return out
 
 
 
 
 
class ResNet50(BasicModule):
    def __init__(self,  layers = [3, 4, 6, 3], numclass = 2):
        self.inplanes = 64
        super(ResNet50, self).__init__() ## super函数是用于调用父类(超类)的一个方法
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)   ##inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(residual_block, 64, blocks = layers[0], stride=1)
        self.layer2 = self._make_layer(residual_block, 128, blocks = layers[1], stride=2)
        self.layer3 = self._make_layer(residual_block, 256, blocks = layers[2], stride=2)
        self.layer4 = self._make_layer(residual_block, 512, blocks = layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512* residual_block.expansion, numclass)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride !=1 or self.inplanes != block.expansion * planes :
            print (planes, blocks)
            ## torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) ###该部分是将每个blocks的第一个residual结构保存在layers列表中,这个地方是用来进行下采样的
        self.inplanes = planes * block.expansion
 
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))##该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        bn1 = self.bn1(x)
        relu = self.relu(bn1)
        maxpool = self.maxpool(relu)
        layer1 = self.layer1(maxpool)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
 
        avgpool = self.avgpool(layer4)
        x = avgpool.view(avgpool.size(0),-1)
        x = self.fc(x)
 
        return x
