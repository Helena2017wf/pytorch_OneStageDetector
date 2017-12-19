# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import argparse

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import CLASSES as labels
from matplotlib import pyplot as plt
from data import GetDataset, DatasetRoot, AnnotationTransform
from Nets import get_net,get_config
from evaluation import Timer
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--net', default='PDN', help='detection network')
parser.add_argument('--input_dim', default='512', help='the dimension of the input image')
parser.add_argument('--trained_model', default='weights/PDN512_Caltech_visible.pth',
                    type=str, help='Trained state_dict file path to open')

args = parser.parse_args()
# from models import build_ssd as build_ssd_v1 # uncomment for older pool6 model
num_classes = len(labels)+1
input_dim = int(args.input_dim)
cfg = get_config(args.net+args.input_dim)
net_class = get_net(args.net)
net = net_class(input_dim,'test', num_classes, cfg)    # initialize SSD

net.load_weights(args.trained_model)


testset = GetDataset(DatasetRoot, None, AnnotationTransform(),dataset_name='test01')

for index in range(1000,testset.num_samples):
    # if i%10 != 0.0:
    #     continue
    _t = Timer()
    _t.tic()
    image = testset.pull_image(index)
    img_height,img_width = image.shape[:2]
    _,anno = testset.pull_anno(index,img_width,img_height)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform


    x = cv2.resize(image, (input_dim, input_dim)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()

    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    top_k=10
    # plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.2:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1
        currentAxis.text(10, 10, "{}".format(index+1))
    for k in anno:
        coords = (int(k[0]*img_width),int(k[1]*img_height)) , int((k[2]-k[0])*img_width),int((k[3]-k[1])*img_height)
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=2))
    print("Time:",_t.toc())
    plt.pause(0.01)
    plt.draw()
    plt.gcf().clear()