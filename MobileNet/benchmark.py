import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
from mobile18 import MobileNet18

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x




"""
class MobileNet18(nn.Module):
    def __init__(self):
        super(MobileNet18, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_res(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 7, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_res(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 7, stride, 3, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_res(3, 64, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_dw(64, 64, 2),
            conv_dw(64, 64, 2),
            conv_dw(64, 64, 2),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 2),
            conv_dw(128, 128, 2),
            conv_dw(128, 128, 2),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 2),
            conv_dw(256, 256, 2),
            conv_dw(256, 256, 2),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 2),
            conv_dw(512, 512, 2),
            conv_dw(512, 512, 2),
            conv_dw(512, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
"""
def speed(model, name):
    t0 = time.time()
    input = torch.rand(1,3,224,224).cuda()
    input = Variable(input, volatile = True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()
    
    print('%10s : %f' % (name, t3 - t2))

def num_para(model, name):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('%10s : %f' % (name, pytorch_total_params))

if __name__ == '__main__':
    #cudnn.benchmark = True # This will make network slow ??
    resnet18 = models.resnet18().cuda()
    alexnet = models.alexnet().cuda()
    vgg16 = models.vgg16().cuda()
    squeezenet = models.squeezenet1_0().cuda()
    mobilenet = MobileNet().cuda()
    mobilenet18 = MobileNet18().cuda()

    speed(resnet18, 'resnet18')
    speed(alexnet, 'alexnet')
    speed(vgg16, 'vgg16')
    speed(squeezenet, 'squeezenet')
    speed(mobilenet, 'mobilenet')
    speed(mobilenet18, 'mobilenet18')

    num_para(resnet18, 'resnet18')
    num_para(alexnet, 'alexnet')
    num_para(vgg16, 'vgg16')
    num_para(squeezenet, 'squeezenet')
    num_para(mobilenet, 'mobilenet')
    num_para(mobilenet18, 'mobilenet18')
