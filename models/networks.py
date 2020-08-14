import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F


### help functions ###
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        lambda_rule = lambda epoch: opt.lr_gamma ** ((epoch+1) // opt.lr_decay_epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
        net = net
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'pretrained':
                    pass
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
        print('initialize network with %s' % init_type)
        net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    for root_child in net.children():
        for children in root_child.children():
            if children in root_child.need_initialization:
                init_weights(children, init_type, gain=init_gain)
            else:
                init_weights(children, "pretrained", gain=init_gain)
    return net

def define_AAUNet(num_labels, init_type='xavier', init_gain=0.02, gpu_ids=[]):

    net = AAUNet(num_labels)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_AARTFNet(num_labels, init_type='xavier', init_gain=0.02, gpu_ids=[]):

    net = AARTFNet(num_labels)
    return init_net(net, init_type, init_gain, gpu_ids)


### Three attention modules ###
class PAM(nn.Module):
    """ Position Attention Module """
    def __init__(self, channel):
        super(PAM, self).__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        _, c, _, _ = x.size()
        y = self.act(self.conv(x))
        y = y.repeat(1, c, 1, 1)
        return x * y


class CAM(nn.Module):
    """ Channel Attention Module """
    def __init__(self, channel, reduction=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DAM_Position(nn.Module):
    """ Position attention submodule in Dual Attention Module"""
    # Ref from https://github.com/junfu1115/DANet
    def __init__(self, in_dim):
        super(DAM_Position, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DAM_Channel(nn.Module):
    """ Channel attention submodule in Dual Attention Module """
    # Ref from https://github.com/junfu1115/DANet
    def __init__(self, in_dim):
        super(DAM_Channel, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


### AA-UNet ###
# Developed based on U-Net provided in https://github.com/yassouali/pytorch_segmentation
class AAUNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, **_):
        super(AAUNet, self).__init__()
        self.down1 = encoder(in_channels, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        self.down4 = encoder(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # attention module
        self.dam_p = DAM_Position(1024)
        self.dam_c = DAM_Channel(1024)
        # self.cam1 = CAM(64)
        self.cam2 = CAM(128)
        self.cam3 = CAM(256)
        self.cam4 = CAM(512)

        self.pam1 = PAM(64)
        # self.pam2 = PAM(128)
        # self.pam3 = PAM(256)
        # self.pam4 = PAM(512)

        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # initialization layers
        self.need_initialization = [self.down1, self.down2, self.down3, self.down4, self.middle_conv,
                                    self.dam_p, self.dam_c, self.cam2, self.cam3, self.cam4, self.pam1,
                                    self.up1, self.up2, self.up3, self.up4, self.final_conv]

    def forward(self, x):
        # encoder
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.middle_conv(x)

        # attention module
        x = self.dam_p(x) + self.dam_c(x)
        x4 = self.cam4(x4)
        x3 = self.cam3(x3)
        x2 = self.cam2(x2)
        x1 = self.pam1(x1)

        # decoder
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)
        return x


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_copy, x):
        x = self.up(x)
        # Padding in case the incomping volumes are of different sizes
        diffY = x_copy.size()[2] - x.size()[2]
        diffX = x_copy.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


### AA-RTFNet ###
# Developed based on RTFNet provided in https://github.com/yuxiangsun/RTFNet
class AARTFNet(nn.Module):
    def __init__(self, num_classes, rgb_enc=True, depth_enc=True, rgb_dec=True, depth_dec=False):
        super(AARTFNet, self).__init__()

        self.need_initialization = []   # we use the initialization code provided in RTFNet
        self.num_resnet_layers = 50

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = torchvision.models.resnet18(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = torchvision.models.resnet34(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = torchvision.models.resnet50(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = torchvision.models.resnet101(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = torchvision.models.resnet152(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet152(pretrained=True)
            self.inplanes = 2048

        # tdisp encoder
        # construct the tdisp encoder, and copy the weights at the same time
        self.encoder_tdisp_conv1 = resnet_raw_model1.conv1
        self.encoder_tdisp_bn1 = resnet_raw_model1.bn1
        self.encoder_tdisp_relu = resnet_raw_model1.relu
        self.encoder_tdisp_maxpool = resnet_raw_model1.maxpool
        self.encoder_tdisp_layer1 = resnet_raw_model1.layer1
        self.encoder_tdisp_layer2 = resnet_raw_model1.layer2
        self.encoder_tdisp_layer3 = resnet_raw_model1.layer3
        self.encoder_tdisp_layer4 = resnet_raw_model1.layer4

        # rgb encoder
        # construct the rgb encoder, and copy the weights at the same time
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        # attention module
        self.dam_p = DAM_Position(2048)
        self.dam_c = DAM_Channel(2048)
        # self.cam1 = CAM(64)
        self.cam2 = CAM(256)
        self.cam3 = CAM(512)
        self.cam4 = CAM(1024)

        self.pam1 = PAM(64)
        # self.pam2 = PAM(256)
        # self.pam3 = PAM(512)
        # self.pam4 = PAM(1024)

        # decoder
        self.deconv1 = self._make_transpose_layer(TransBottleneck, int(self.inplanes/2), 2, stride=2, input_channel=1)
        self.deconv2 = self._make_transpose_layer(TransBottleneck, int(self.inplanes/2), 2, stride=2, input_channel=2)
        self.deconv3 = self._make_transpose_layer(TransBottleneck, int(self.inplanes/2), 2, stride=2, input_channel=2)
        self.deconv4 = self._make_transpose_layer(TransBottleneck, int(self.inplanes/4), 2, stride=2, input_channel=2)
        self.deconv5 = self._make_transpose_layer(TransBottleneck, num_classes, 2, stride=2, input_channel=2)

    def _make_transpose_layer(self, block, planes, blocks, stride=1, input_channel=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            ) # if stride =2, double the resolution
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            ) # if run this branch, stride must be 1, so this statement will keep the resolution.

        # initialize the weight of upsample layer
        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes * input_channel, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, rgb, tdisp):
        # encoder
        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)
        tdisp = self.encoder_tdisp_conv1(tdisp)
        tdisp = self.encoder_tdisp_bn1(tdisp)
        tdisp = self.encoder_tdisp_relu(tdisp)
        x1 = rgb + tdisp

        rgb = self.encoder_rgb_maxpool(x1)
        tdisp = self.encoder_tdisp_maxpool(tdisp)
        rgb = self.encoder_rgb_layer1(rgb)
        tdisp = self.encoder_tdisp_layer1(tdisp)
        x2 = rgb + tdisp

        rgb = self.encoder_rgb_layer2(x2)
        tdisp = self.encoder_tdisp_layer2(tdisp)
        x3 = rgb + tdisp

        rgb = self.encoder_rgb_layer3(x3)
        tdisp = self.encoder_tdisp_layer3(tdisp)
        x4 = rgb + tdisp

        rgb = self.encoder_rgb_layer4(x4)
        tdisp = self.encoder_tdisp_layer4(tdisp)
        x = rgb + tdisp

        # attention module
        x = self.dam_p(x) + self.dam_c(x)
        x4 = self.cam4(x4)
        x3 = self.cam3(x3)
        x2 = self.cam2(x2)
        x1 = self.pam1(x1)

        # decoder
        x = self.deconv1(x)
        x = self.deconv2(torch.cat([x4, x], dim=1))
        x = self.deconv3(torch.cat([x3, x], dim=1))
        x = self.deconv4(torch.cat([x2, x], dim=1))
        x = self.deconv5(torch.cat([x1, x], dim=1))

        out = x
        return out


class TransBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # keep resolution
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # keep resolution if stride=1
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False) # double the resolution if stride=2
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # keep resolution if stride=1

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

        # initialize the weight of each layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        residual = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class SegmantationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(SegmantationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
    def __call__(self, output, target, pixel_average=True):
        if pixel_average:
            return self.loss(output, target) #/ target.data.sum()
        else:
            return self.loss(output, target)
