import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os


class PDN(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, size ,phase, num_classes,cfg):
        super(PDN, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.config = cfg
        self.priorbox = PriorBox(self.config)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # network: base, extras, head
        self._vgg = self.vgg(self.config['base'], 3)
        self._extras = None
      #  self._extras = self.add_extras(self.config['extra'], 1024)
        self._loc, self._conf = self.multibox(self.config['mbox'], num_classes)
        # Layer learns to scale the l2 normalized features from conv4_3 and conv3_3
        self.L2Norm_conv3 = L2Norm(256, 10)
        self.L2Norm_conv4 = L2Norm(512,8)
        self.L2Norm_conv5 = L2Norm (512,5)
      ########### modulelist ##########
        self._vgg = nn.ModuleList(self._vgg)
       # self._extras = nn.ModuleList(self._extras)
        self._loc = nn.ModuleList(self._loc)
        self._conf = nn.ModuleList(self._conf)

        self.softmax = nn.Softmax()
        # self.detect = Detect(num_classes, 0, 200, 0.01, 0.45) ##ssd original
        self.detect = Detect(num_classes,cfg, 0, 200, 0.01, 0.45) ##for Kaist

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv3_3 relu
        for k in range(16):
            x = self._vgg[k](x)
        s = self.L2Norm_conv3(x)
        sources.append(s)

        # apply vgg up to conv4_3 relu
        for k in range(16,23):
            x = self._vgg[k](x)
        s = self.L2Norm_conv4(x)
        sources.append(s)

        # apply vgg up to conv4_3 relu
        for k in range(23,30):
            x = self._vgg[k](x)
        s = self.L2Norm_conv5(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self._vgg)):
            x = self._vgg[k](x)
        sources.append(x)

        # # apply extra layers and cache source layer outputs
        # for k, v in enumerate(self._extras):
        #     x = F.relu(v(x), inplace=True)
        #     if k % 2 == 1:
        #         sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self._loc, self._conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    # This function is derived from torchvision VGG make_layers()
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    def vgg(self,cfg, i, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # elif v == 'M3':
            #     layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        pool7 = nn.MaxPool2d(kernel_size=2, stride=2)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True),pool7]
        return layers

    def add_extras(self,cfg, i, batch_norm=False):
        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = i
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'L':
                    layers += [nn.Conv2d(in_channels, 256,
                                         kernel_size=4, padding=1)]
                elif v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        return layers


    def multibox(self,cfg, num_classes):
        loc_layers = []
        conf_layers = []
        vgg_source = [14,21,28,33] #conv3,4,5,7
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(self._vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self._vgg[v].out_channels,
                            cfg[k] * num_classes, kernel_size=3, padding=1)]
        # for k, v in enumerate(self._extra_layers[1::2], 2):
        #     loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
        #                              * 4, kernel_size=3, padding=1)]
        #     conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
        #                               * num_classes, kernel_size=3, padding=1)]
        return loc_layers, conf_layers

    def set_phase(self, phase):
        assert phase in ['train', 'test']
        self.phase = phase